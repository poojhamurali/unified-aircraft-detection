"""Which changes make the sensor router more accurate and more robust?

Same images as stat_router.py (400 train / up to 400 test per sensor). Compared:

    A  8 statistics + Gaussian (QDA)                 baseline from stat_router.py
    B  8 statistics + random forest
    C  8 statistics + gradient boosting
    D  extended statistics (+histogram, percentiles, channel equality,
       frequency spectrum, gradients, skew/kurtosis) + gradient boosting
    E  tiny CNN on 128x128 thumbnails (learns its own features)
    F  D and E averaged (ensemble)

Then for the best router: accuracy when it only routes images it is confident
about (the rest go to a fallback), and robustness when test images are perturbed
(grayscale optical, brightness change, sensor noise, heavy JPEG).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from unified_yolo import MODALITIES, image_statistics  # noqa: E402

OUT = HERE / "outputs" / "stat_router"
IMGSZ, THUMB = 640, 128
torch.manual_seed(0)
np.random.seed(0)


def files_for(split, per_modality=400):
    out = []
    for m, mod in enumerate(MODALITIES):
        imgs = sorted((HERE / "datasets/unified" / mod / "images" / split).iterdir())
        step = max(1, len(imgs) // per_modality)
        out += [(p, m) for p in imgs[::step][:per_modality]]
    return out


def read(p):
    im = cv2.imread(str(p))
    return cv2.resize(im, (IMGSZ, IMGSZ), interpolation=cv2.INTER_AREA)[:, :, ::-1].copy()


def to_tensor(im):
    return torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1)[None].float() / 255


def extended_stats(im_rgb):
    """Extra hand-crafted features beyond the model's 8 statistics."""
    x = im_rgb.astype(np.float32) / 255
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    feats = []
    feats += list(np.histogram(gray, bins=16, range=(0, 1))[0] / gray.size)        # 16 histogram bins
    feats += list(np.percentile(gray, [5, 25, 50, 75, 95]))                        # 5 percentiles
    feats += [np.abs(r - g).mean(), np.abs(g - b).mean(), np.abs(r - b).mean(),   # channel differences
              np.mean((np.abs(r - g) < 1e-3) & (np.abs(g - b) < 1e-3))]           # exact-gray fraction
    small = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    spec = np.abs(np.fft.fftshift(np.fft.fft2(small)))
    yy, xx = np.mgrid[-64:64, -64:64]
    rad = np.sqrt(xx ** 2 + yy ** 2)
    tot = spec.sum() + 1e-8
    feats += [spec[rad < 8].sum() / tot, spec[(rad >= 8) & (rad < 32)].sum() / tot, spec[rad >= 32].sum() / tot]
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    feats += [mag.mean(), mag.std(), np.mean(mag > 0.2)]
    z = (gray - gray.mean()) / (gray.std() + 1e-6)
    feats += [np.mean(z ** 3), np.log1p(np.mean(z ** 4))]                          # skew, kurtosis
    return np.array(feats, dtype=np.float32)


def featurize(items, perturb=None):
    base, ext, thumbs, y = [], [], [], []
    for p, m in items:
        im = read(p)
        if perturb is not None:
            im = perturb(im, m)
        base.append(image_statistics(to_tensor(im))[0].numpy())
        ext.append(extended_stats(im))
        thumbs.append(cv2.resize(im, (THUMB, THUMB), interpolation=cv2.INTER_AREA))
        y.append(m)
    base, ext = np.stack(base), np.stack(ext)
    return np.concatenate([base, ext], 1), base, np.stack(thumbs), np.array(y)


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.SiLU(), nn.MaxPool2d(2))
        self.f = nn.Sequential(blk(3, 16), blk(16, 32), blk(32, 64), blk(64, 64), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.h = nn.Linear(64, len(MODALITIES))

    def forward(self, x):
        return self.h(self.f(x))


def thumbs_tensor(t):
    return torch.from_numpy(t).permute(0, 3, 1, 2).float() / 255


def train_cnn(T, y, epochs=25):
    net = TinyCNN()
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    X, Y = thumbs_tensor(T), torch.from_numpy(y).long()
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 32):
            idx = perm[i:i + 32]
            xb = X[idx]
            if torch.rand(1) < 0.5:
                xb = xb.flip(3)
            xb = (xb * (0.8 + 0.4 * torch.rand(len(xb), 1, 1, 1))).clamp(0, 1)  # brightness jitter
            loss = F.cross_entropy(net(xb), Y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return net.eval()


def cnn_proba(net, T):
    with torch.no_grad():
        return net(thumbs_tensor(T)).softmax(1).numpy()


def summarize(y, p):
    c = np.zeros((3, 3), int)
    for a, b in zip(y, p):
        c[a, b] += 1
    return {"accuracy": round(float((y == p).mean()), 4),
            "recall": {mod: round(float(c[m, m] / max(c[m].sum(), 1)), 4) for m, mod in enumerate(MODALITIES)},
            "confusion": c.tolist(), "errors": int((y != p).sum()), "n": int(len(y))}


# ---- perturbations for the robustness test ---------------------------------- #
def p_gray_optical(im, m):
    if m != 0:
        return im
    g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    return np.repeat(g[..., None], 3, 2)


def p_bright(im, m):
    return np.clip(im.astype(np.float32) * 1.4 + 20, 0, 255).astype(np.uint8)


def p_noise(im, m):
    rng = np.random.default_rng(0)
    return np.clip(im.astype(np.float32) + rng.normal(0, 12, im.shape), 0, 255).astype(np.uint8)


def p_jpeg(im, m):
    ok, buf = cv2.imencode(".jpg", im[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 15])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)[:, :, ::-1].copy()


def main():
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

    tr, te = files_for("train"), files_for("test")
    print(f"featurizing {len(tr)} train / {len(te)} test images...")
    Xtr, Btr, Ttr, ytr = featurize(tr)
    Xte, Bte, Tte, yte = featurize(te)

    res = {}
    models = {
        "A_8stats_gaussian": (QuadraticDiscriminantAnalysis(reg_param=0.05), "base"),
        "B_8stats_random_forest": (RandomForestClassifier(400, random_state=0), "base"),
        "C_8stats_gradient_boosting": (HistGradientBoostingClassifier(random_state=0), "base"),
        "D_extended_stats_gradient_boosting": (HistGradientBoostingClassifier(random_state=0), "ext"),
    }
    probas = {}
    for name, (clf, which) in models.items():
        A, B = (Btr, Bte) if which == "base" else (Xtr, Xte)
        clf.fit(A, ytr)
        probas[name] = clf.predict_proba(B)
        res[name] = summarize(yte, probas[name].argmax(1))
        print(name, res[name]["accuracy"], res[name]["errors"], "errors")

    print("training tiny CNN...")
    net = train_cnn(Ttr, ytr)
    probas["E_tiny_cnn"] = cnn_proba(net, Tte)
    res["E_tiny_cnn"] = summarize(yte, probas["E_tiny_cnn"].argmax(1))
    print("E_tiny_cnn", res["E_tiny_cnn"]["accuracy"])

    probas["F_ensemble_D_plus_E"] = (probas["D_extended_stats_gradient_boosting"] + probas["E_tiny_cnn"]) / 2
    res["F_ensemble_D_plus_E"] = summarize(yte, probas["F_ensemble_D_plus_E"].argmax(1))
    print("F_ensemble", res["F_ensemble_D_plus_E"]["accuracy"])

    best = max(res, key=lambda k: res[k]["accuracy"])
    pb = probas[best]
    conf, pred = pb.max(1), pb.argmax(1)
    coverage = []
    for t in (0.5, 0.7, 0.8, 0.9, 0.95, 0.99):
        keep = conf >= t
        coverage.append({"threshold": t, "routed_fraction": round(float(keep.mean()), 4),
                         "accuracy_on_routed": round(float((pred[keep] == yte[keep]).mean()) if keep.any() else 0, 4),
                         "sent_to_fallback": int((~keep).sum())})

    # robustness: re-featurize perturbed test images, score the stats router (D) and CNN (E)
    clf_d = models["D_extended_stats_gradient_boosting"][0]
    robust = {}
    for tag, fn in (("optical_converted_to_grayscale", p_gray_optical), ("brighter_x1.4", p_bright),
                    ("gaussian_noise_sigma12", p_noise), ("jpeg_quality_15", p_jpeg)):
        Xp, _, Tp, yp = featurize(te, fn)
        pd_, pe = clf_d.predict_proba(Xp), cnn_proba(net, Tp)
        robust[tag] = {"D_stats": summarize(yp, pd_.argmax(1))["accuracy"],
                       "E_cnn": summarize(yp, pe.argmax(1))["accuracy"],
                       "F_ensemble": summarize(yp, ((pd_ + pe) / 2).argmax(1))["accuracy"]}
        print("robust", tag, robust[tag])

    report = {"routers": res, "best": best, "confidence_routing_for_best": coverage, "robustness": robust}
    (OUT / "router_improvements.json").write_text(json.dumps(report, indent=2))
    torch.save(net.state_dict(), OUT / "tiny_cnn_router.pt")
    print(json.dumps({k: (v["accuracy"], v["errors"]) for k, v in res.items()}, indent=2))
    print(json.dumps(coverage, indent=2))
    print(f"\n-> {OUT / 'router_improvements.json'}")


if __name__ == "__main__":
    main()
