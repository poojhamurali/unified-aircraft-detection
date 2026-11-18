"""Quality- and modality-conditioned YOLO.

One shared YOLO detector whose internal features are steered, per image, by FiLM
(feature-wise linear modulation) parameters computed from two things:

    Image --> QualityNet --> quality vector --+
                                              +--> MLP --> cond --> FiLM(gamma, beta)
    Modality id --> modality token -----------+                       at chosen layers

* QualityNet looks at the raw image only. It combines a handful of measurable
  statistics (contrast, dynamic range, noise/speckle, colourfulness, target
  conspicuity, dark fraction) with a tiny CNN, and also predicts which sensor the
  image came from.
* The modality token is the *true* sensor id while training (teacher forcing) and
  the QualityNet's own prediction at test time, so the deployed model needs no
  sensor label.
* Every FiLM layer is zero-initialised, so training starts from exactly the
  pretrained single-sensor YOLO and only departs from it where it helps.

``cond_mode`` switches parts off for the ablation table:
    full           quality vector + modality token   (proposed)
    modality_only  modality token only
    quality_only   quality vector only
    off            no conditioning at all (plain pooled YOLO baseline)

FiLM itself is not new; what is studied here is its use as a sensor- and
quality-aware adapter for a single detector trained on unpaired optical, infrared
and SAR data.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER

from .modality import MODALITIES, modality_of

N_STATS = 8
COND_MODES = ("full", "modality_only", "quality_only", "off")


# --------------------------------------------------------------------------- #
# Image-quality analysis
# --------------------------------------------------------------------------- #
@torch.no_grad()
def image_statistics(img: torch.Tensor) -> torch.Tensor:
    """Eight measurable image-quality statistics, shape (B, 8), roughly in [0, 2].

    ``img`` is (B, 3, H, W) in [0, 1]. Statistics that depend on fine detail
    (noise, speckle) are computed at native resolution, because downsampling
    first would average exactly that detail away.
    """
    x = img.float()
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    gray = 0.299 * r + 0.587 * g + 0.114 * b  # (B,1,H,W)

    mean = gray.mean(dim=(1, 2, 3))
    std = gray.std(dim=(1, 2, 3))

    # dynamic range: 1st-99th percentile on a 64x64 summary
    if gray.device.type == "mps" and (gray.shape[-2] % 64 or gray.shape[-1] % 64):
        # MPS adaptive pooling only supports sizes divisible by the output size,
        # and letterboxed validation batches (e.g. 640x352) are not
        small = F.adaptive_avg_pool2d(gray.cpu(), 64).to(gray.device)
    else:
        small = F.adaptive_avg_pool2d(gray, 64)
    small = small.flatten(1)
    s, _ = small.sort(dim=1)
    n = s.shape[1]
    dyn = s[:, int(0.99 * (n - 1))] - s[:, int(0.01 * (n - 1))]

    # high-frequency energy: mean |Laplacian|
    k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], device=x.device).view(1, 1, 3, 3)
    lap = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="replicate"), k).abs().mean(dim=(1, 2, 3))
    lap = torch.log1p(60.0 * lap) / 2.0

    # speckle index: local std / local mean, averaged over the frame
    gp = F.pad(gray, (2, 2, 2, 2), mode="replicate")
    m1 = F.avg_pool2d(gp, 5, stride=1)
    m2 = F.avg_pool2d(gp * gp, 5, stride=1)
    local_std = (m2 - m1 * m1).clamp_min(0).sqrt()
    speckle = (local_std / (m1 + 0.02)).clamp(max=3.0).mean(dim=(1, 2, 3)) / 3.0

    # colourfulness: ~0 for single-band sensors replicated into 3 channels
    color = ((r - g).abs() + (g - b).abs()).mean(dim=(1, 2, 3)) * 4.0

    # target conspicuity: strongest local excess over the neighbourhood, in std units
    bg = F.avg_pool2d(F.pad(gray, (7, 7, 7, 7), mode="replicate"), 15, stride=1)
    peak = (gray - bg).amax(dim=(1, 2, 3)) / (std + 1e-2)
    peak = torch.log1p(peak) / 3.0

    dark = (gray < 0.05).float().mean(dim=(1, 2, 3))

    return torch.stack([mean, std, dyn, lap, speckle, color, peak, dark], dim=1)


class QualityNet(nn.Module):
    """Image -> (quality vector, modality logits)."""

    def __init__(self, qdim: int = 32, n_mod: int = len(MODALITIES)):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1), nn.BatchNorm2d(16), nn.SiLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.SiLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.stats = nn.Sequential(nn.Linear(N_STATS, 32), nn.SiLU())
        self.fuse = nn.Sequential(nn.Linear(64 + 32, qdim), nn.SiLU())
        self.modality_head = nn.Linear(qdim, n_mod)

    def forward(self, img: torch.Tensor):
        x = img.float()
        small = F.interpolate(x, size=(128, 128), mode="bilinear", align_corners=False)
        feat = torch.cat([self.cnn(small), self.stats(image_statistics(x))], dim=1)
        q = self.fuse(feat)
        return q, self.modality_head(q)


# --------------------------------------------------------------------------- #
# FiLM wrapper around a YOLO layer
# --------------------------------------------------------------------------- #
class FiLMLayer(nn.Module):
    """Wrap one YOLO layer: ``out = layer(x) * (1 + tanh(gamma)) + beta``.

    The wrapper mirrors the attributes the Ultralytics forward loop reads
    (``f``, ``i``, ``type``, ``np``) so it drops into ``model.model`` unchanged.
    ``cond`` is set by the parent model just before each forward pass and cleared
    right after, so no graph-carrying tensor is left on the module (which would
    break ``deepcopy`` in the EMA).
    """

    def __init__(self, layer: nn.Module, channels: int, cond_dim: int):
        super().__init__()
        self.layer = layer
        self.f, self.i = layer.f, layer.i
        self.type, self.np = layer.type, getattr(layer, "np", 0)
        self.channels = channels
        self.to_gamma_beta = nn.Linear(cond_dim, 2 * channels)
        nn.init.zeros_(self.to_gamma_beta.weight)
        nn.init.zeros_(self.to_gamma_beta.bias)
        self.cond = None

    def forward(self, x):
        y = self.layer(x)
        if self.cond is None:
            return y
        gamma, beta = self.to_gamma_beta(self.cond).chunk(2, dim=1)
        gamma = torch.tanh(gamma).to(y.dtype)[:, :, None, None]
        beta = beta.to(y.dtype)[:, :, None, None]
        return y * (1.0 + gamma) + beta


# --------------------------------------------------------------------------- #
# The detector
# --------------------------------------------------------------------------- #
class UnifiedDetectionModel(DetectionModel):
    """YOLO detector with quality- and modality-conditioned FiLM adapters."""

    def __init__(self, cfg="yolov8s.yaml", ch=3, nc=None, verbose=True,
                 film_layers=(2, 4, 6, 9, 12, 15, 18, 21), cond_mode="full",
                 qdim=32, cond_dim=128, aux_weight=0.1):
        super().__init__(cfg, ch=ch, nc=nc, verbose=verbose)
        if cond_mode not in COND_MODES:
            raise ValueError(f"cond_mode must be one of {COND_MODES}")
        self.film_layers = tuple(film_layers)
        self.cond_mode = cond_mode
        self.qdim, self.cond_dim, self.aux_weight = qdim, cond_dim, aux_weight

        self.quality = QualityNet(qdim)
        self.modality_embedding = nn.Embedding(len(MODALITIES), qdim)
        self.cond_mlp = nn.Sequential(nn.Linear(2 * qdim, cond_dim), nn.SiLU())

        channels = self._layer_channels(self.film_layers)
        self.films = []
        for idx in self.film_layers:
            wrapped = FiLMLayer(self.model[idx], channels[idx], cond_dim)
            self.model[idx] = wrapped
            self.films.append(idx)

        self._true_ids = None      # set by loss(); teacher-forced modality ids
        self._forced_ids = None    # optional manual override (ablations)
        self._aux_logits = None
        self.last_modality_probs = None

    # -- construction helpers ------------------------------------------------ #
    @torch.no_grad()
    def _layer_channels(self, indices):
        """Output channel count of the requested layers, via a throwaway forward pass."""
        sizes, hooks = {}, []
        for idx in indices:
            def hook(_m, _i, out, idx=idx):
                sizes[idx] = out.shape[1]
            hooks.append(self.model[idx].register_forward_hook(hook))
        was_training = self.training
        self.eval()
        p = next(self.parameters())
        DetectionModel._predict_once(self, torch.zeros(1, 3, 64, 64, device=p.device, dtype=p.dtype))
        self.train(was_training)
        for h in hooks:
            h.remove()
        return sizes

    def load(self, weights, verbose=True):
        """Load plain-YOLO or unified weights; wrapped layers get ``.layer.`` inserted."""
        model = weights["model"] if isinstance(weights, dict) else weights
        csd = model.float().state_dict()
        own = self.state_dict()
        remapped = {}
        for k, v in csd.items():
            if k in own:
                remapped[k] = v
                continue
            for idx in self.film_layers:
                pre = f"model.{idx}."
                if k.startswith(pre):
                    nk = f"model.{idx}.layer." + k[len(pre):]
                    if nk in own:
                        remapped[nk] = v
                    break
        # keep only shape-compatible tensors (the 80-class COCO head is dropped)
        remapped = {k: v for k, v in remapped.items() if own[k].shape == v.shape}
        self.load_state_dict(remapped, strict=False)
        if verbose:
            LOGGER.info(f"Transferred {len(remapped)}/{len(own)} items into the unified detector")

    # -- conditioning -------------------------------------------------------- #
    def set_modality(self, ids):
        """Force the modality ids (tensor of shape (B,)) or ``None`` to predict them."""
        self._forced_ids = ids

    def _compute_cond(self, x):
        q, logits = self.quality(x)
        self._aux_logits = logits
        probs = logits.softmax(dim=1)
        self.last_modality_probs = probs.detach()

        ids = self._forced_ids if self._forced_ids is not None else (
            self._true_ids if self.training else None)
        if ids is not None:
            token = self.modality_embedding(ids.to(x.device))
        else:  # deployment: the model's own belief about which sensor this is
            token = probs @ self.modality_embedding.weight

        if self.cond_mode == "modality_only":
            q = torch.zeros_like(q)
        elif self.cond_mode == "quality_only":
            token = torch.zeros_like(token)
        return self.cond_mlp(torch.cat([q, token], dim=1))

    def _predict_once(self, x, *args, **kwargs):
        # during DetectionModel.__init__ (stride probe) nothing is wrapped yet
        if getattr(self, "cond_mode", "off") == "off" or not getattr(self, "films", None):
            return super()._predict_once(x, *args, **kwargs)
        cond = self._compute_cond(x)
        for idx in self.films:
            self.model[idx].cond = cond
        try:
            return super()._predict_once(x, *args, **kwargs)
        finally:
            for idx in self.films:
                self.model[idx].cond = None

    # -- training loss ------------------------------------------------------- #
    def loss(self, batch, preds=None):
        device = batch["img"].device
        ids = torch.tensor([modality_of(p) for p in batch["im_file"]], device=device)
        self._true_ids = ids
        try:
            total, items = super().loss(batch, preds)
        finally:
            self._true_ids = None
        logits, self._aux_logits = self._aux_logits, None  # never keep a graph on the module
        if self.cond_mode != "off" and logits is not None:
            aux = F.cross_entropy(logits, ids)
            bs = batch["img"].shape[0]
            extra = self.aux_weight * aux * bs
            total = total + extra if total.ndim == 0 else total + extra / total.numel()
        return total, items

    # -- diagnostics --------------------------------------------------------- #
    def film_report(self):
        """Mean |gamma| weight norm per FiLM layer: shows where adaptation was learned."""
        return {idx: float(self.model[idx].to_gamma_beta.weight.abs().mean())
                for idx in self.films}
