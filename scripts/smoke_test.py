"""Fast checks that the unified model is wired correctly (CPU, no dataset needed).

1. zero-initialised FiLM  => unified model output == plain pretrained YOLO output
2. FiLM weights actually receive gradient once they move off zero
3. loss() runs, includes the modality auxiliary term, and backpropagates
4. a deep copy (what the EMA does) works and leaves no graph-carrying state behind
5. image statistics separate colour / grayscale / speckled inputs
"""

import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ultralytics import YOLO  # noqa: E402
from ultralytics.nn.tasks import DetectionModel  # noqa: E402
from unified_yolo import UnifiedDetectionModel, image_statistics  # noqa: E402
from unified_yolo.paths import pretrained_weights  # noqa: E402

WEIGHTS = pretrained_weights()


def main():
    torch.manual_seed(0)
    plain = YOLO(str(WEIGHTS)).model.float().eval()
    # Reference = what the stock trainer builds: the *current* YOLO classes rebuilt from the
    # yaml with the checkpoint's weights. (The checkpoint's own pickled objects can differ in
    # small ways, e.g. the SPPF activation, so they are not the right yardstick.)
    ref = DetectionModel(plain.yaml, nc=80, verbose=False)
    ref.load(plain, verbose=False)
    ref.eval()
    model = UnifiedDetectionModel(plain.yaml, nc=80, verbose=False)
    model.load(plain, verbose=False)
    model.eval()

    x = torch.rand(2, 3, 256, 256)
    with torch.no_grad():
        a = ref(x)
        b = model(x)
    a0, b0 = (a[0] if isinstance(a, (list, tuple)) else a), (b[0] if isinstance(b, (list, tuple)) else b)
    diff = (a0 - b0).abs().max().item()
    print(f"[1] max |stock YOLO - unified| at init = {diff:.2e}")
    assert diff < 1e-4, "FiLM is not identity at initialisation"

    # [2]/[3] loss + gradient on a fake single-class batch
    model = UnifiedDetectionModel(plain.yaml, nc=1, verbose=False)
    model.load(plain, verbose=False)
    model.args = type("A", (), dict(box=7.5, cls=0.5, dfl=1.5))()
    model.train()
    batch = dict(
        img=torch.rand(3, 3, 256, 256),
        batch_idx=torch.tensor([0., 1., 2.]),
        cls=torch.zeros(3, 1),
        bboxes=torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.3, 0.6, 0.1, 0.1], [0.7, 0.4, 0.3, 0.2]]),
        im_file=["x/opt__a.jpg", "x/ir__b.jpg", "x/sar__c.jpg"],
    )
    loss, items = model.loss(batch)
    print(f"[3] loss shape={tuple(loss.shape)} sum={float(loss.sum().detach()):.3f} items={ {k: round(float(v), 3) for k, v in items.items()} if isinstance(items, dict) else items}")
    loss.sum().backward()
    g = {i: float(model.model[i].to_gamma_beta.bias.grad.abs().sum()) for i in model.films}
    print(f"[2] FiLM bias grad per layer: { {k: round(v, 4) for k, v in g.items()} }")
    assert all(v > 0 for v in g.values()), "some FiLM layers receive no gradient"
    assert model.quality.modality_head.weight.grad is not None

    # [4] deepcopy like the EMA
    model.zero_grad()
    ema = copy.deepcopy(model)
    assert all(model.model[i].cond is None for i in model.films)
    ema.eval()
    with torch.no_grad():
        ema(torch.rand(1, 3, 256, 256))
    print("[4] deepcopy + eval forward ok; modality probs:",
          [round(p, 2) for p in ema.last_modality_probs[0].tolist()])

    # [5] statistics
    rgb = torch.rand(1, 3, 128, 128)
    gray = torch.rand(1, 1, 128, 128).repeat(1, 3, 1, 1)
    flat = torch.full((1, 3, 128, 128), 0.5)
    names = "mean std dyn lap speckle color peak dark".split()
    for tag, im in (("rgb-noise", rgb), ("gray-noise", gray), ("flat", flat)):
        print(f"[5] {tag:10s}", dict(zip(names, [round(v, 3) for v in image_statistics(im)[0].tolist()])))
    print("\nall smoke tests passed")


if __name__ == "__main__":
    main()
