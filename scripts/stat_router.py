"""Route each image to a sensor-specific expert using only its image statistics.

For every image we compute the same 8 measurable statistics the unified model's
QualityNet uses (brightness, contrast, dynamic range, high-frequency detail,
speckle, colourfulness, target conspicuity, dark fraction). A router fitted on
the *train* split learns what each sensor's statistics look like; on the held-out
*test* split it decides, per image, which sensor the statistics resemble most and
sends the image to that sensor's trained detector:

    optical -> Optical/outputs/runs/airport_disjoint_yolov8s_norot/weights/best.pt
    ir      -> IRdata/outputs/runs/top2_yolo11s/weights/best.pt
    sar     -> SAR/outputs/runs/official_yolov8s/weights/best.pt

Two routers are compared:
    nearest-centroid  z-scored distance to each sensor's average statistics
    gaussian          one Gaussian per sensor (QDA); gives a probability per sensor

    python scripts/stat_router.py
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from unified_yolo import MODALITIES, image_statistics  # noqa: E402
from unified_yolo.paths import EXPERTS_ROOT as ROOT  # noqa: E402

STAT_NAMES = ["brightness", "contrast", "dynamic_range", "detail", "speckle",
              "colourfulness", "conspicuity", "dark_fraction"]
EXPERTS = {
    "optical": ROOT / "Optical/outputs/runs/airport_disjoint_yolov8s_norot/weights/best.pt",
    "ir": ROOT / "IRdata/outputs/runs/top2_yolo11s/weights/best.pt",
    "sar": ROOT / "SAR/outputs/runs/official_yolov8s/weights/best.pt",
}
OUT = HERE / "outputs" / "stat_router"
IMGSZ = 640


def load(path: Path) -> torch.Tensor:
    im = cv2.imread(str(path))
    im = cv2.resize(im, (IMGSZ, IMGSZ), interpolation=cv2.INTER_AREA)[:, :, ::-1]
    return torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1)[None].float() / 255


def stats_for(split: str, per_modality: int):
    rows, labels, files = [], [], []
    for m, mod in enumerate(MODALITIES):
        imgs = sorted((HERE / "datasets/unified" / mod / "images" / split).iterdir())
        step = max(1, len(imgs) // per_modality)
        for p in imgs[::step][:per_modality]:
            rows.append(image_statistics(load(p))[0].numpy())
            labels.append(m)
            files.append(str(p))
        print(f"  {split:5s} {mod:8s} {min(len(imgs), per_modality)} images")
    return np.stack(rows), np.array(labels), files


class NearestCentroid:
    def fit(self, X, y):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Z = (X - self.mu) / self.sd
        self.centroids = np.stack([Z[y == m].mean(0) for m in range(len(MODALITIES))])
        return self

    def distances(self, X):
        Z = (X - self.mu) / self.sd
        return np.linalg.norm(Z[:, None, :] - self.centroids[None], axis=2)

    def predict(self, X):
        return self.distances(X).argmin(1)


def confusion(y, p):
    k = len(MODALITIES)
    c = np.zeros((k, k), int)
    for a, b in zip(y, p):
        c[a, b] += 1
    return c


def main():
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

    OUT.mkdir(parents=True, exist_ok=True)
    print("computing statistics...")
    Xtr, ytr, _ = stats_for("train", 400)
    Xte, yte, fte = stats_for("test", 400)

    centroid = NearestCentroid().fit(Xtr, ytr)
    gauss = QuadraticDiscriminantAnalysis(reg_param=0.05).fit(Xtr, ytr)

    report = {"features": STAT_NAMES, "per_modality_train_stats": {}, "routers": {}}
    for m, mod in enumerate(MODALITIES):
        X = Xtr[ytr == m]
        report["per_modality_train_stats"][mod] = {
            n: {"mean": round(float(X[:, i].mean()), 4), "std": round(float(X[:, i].std()), 4)}
            for i, n in enumerate(STAT_NAMES)}

    for name, router in (("nearest_centroid", centroid), ("gaussian", gauss)):
        pred = router.predict(Xte)
        c = confusion(yte, pred)
        report["routers"][name] = {
            "test_accuracy": round(float((pred == yte).mean()), 4),
            "per_modality_recall": {mod: round(float(c[m, m] / c[m].sum()), 4)
                                    for m, mod in enumerate(MODALITIES)},
            "confusion_rows_true_cols_pred": c.tolist(),
        }
        wrong = [(fte[i], MODALITIES[yte[i]], MODALITIES[pred[i]]) for i in np.flatnonzero(pred != yte)]
        report["routers"][name]["misrouted_examples"] = wrong[:10]

    # one worked example per sensor: its statistics, its distance/probability per sensor, its route
    examples = {}
    for m, mod in enumerate(MODALITIES):
        i = int(np.flatnonzero(yte == m)[len(np.flatnonzero(yte == m)) // 2])
        x = Xte[i:i + 1]
        examples[mod] = {
            "file": Path(fte[i]).name,
            "statistics": {n: round(float(v), 4) for n, v in zip(STAT_NAMES, x[0])},
            "distance_to_each_sensor": {k: round(float(d), 3)
                                        for k, d in zip(MODALITIES, centroid.distances(x)[0])},
            "gaussian_probability": {k: round(float(p), 4)
                                     for k, p in zip(MODALITIES, gauss.predict_proba(x)[0])},
            "routed_to": MODALITIES[int(gauss.predict(x)[0])],
            "expert": str(EXPERTS[MODALITIES[int(gauss.predict(x)[0])]].relative_to(ROOT)),
            "_path": fte[i],
        }
    report["examples"] = {k: {kk: vv for kk, vv in v.items() if kk != "_path"} for k, v in examples.items()}

    with open(OUT / "router.pkl", "wb") as f:
        pickle.dump({"gaussian": gauss, "nearest_centroid": centroid, "features": STAT_NAMES,
                     "modalities": MODALITIES, "experts": {k: str(v) for k, v in EXPERTS.items()}}, f)
    np.savez(OUT / "stats.npz", Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
    (OUT / "router_report.json").write_text(json.dumps(report, indent=2))

    plot(Xtr, ytr)
    run_experts(examples)

    print(json.dumps({k: report[k] for k in ("routers", "examples")}, indent=2))
    print(f"\nreport -> {OUT / 'router_report.json'}")


def plot(X, y):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"optical": "#2563EB", "ir": "#DC2626", "sar": "#059669"}
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for i, (ax, name) in enumerate(zip(axes.flat, STAT_NAMES)):
        data = [X[y == m, i] for m in range(len(MODALITIES))]
        bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6)
        for patch, mod in zip(bp["boxes"], MODALITIES):
            patch.set_facecolor(colours[mod])
            patch.set_alpha(0.55)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(["Optical", "Infrared", "SAR"])
        ax.set_title(name.replace("_", " ").title(), fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Image statistics by sensor (train split, 400 images per sensor)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "stats_by_sensor.png", dpi=160)
    plt.close(fig)


def run_experts(examples):
    """Send each example image to the expert it was routed to and save the detections."""
    from ultralytics import YOLO

    for mod, ex in examples.items():
        expert = YOLO(str(ROOT / ex["expert"]))
        res = expert.predict(ex["_path"], imgsz=IMGSZ, conf=0.25, device="cpu", verbose=False)[0]
        ex_out = OUT / f"example_{mod}_routed_to_{ex['routed_to']}.jpg"
        cv2.imwrite(str(ex_out), res.plot())
        ex["detections"] = int(len(res.boxes))
        print(f"  {mod:8s} image -> routed to {ex['routed_to']:8s} expert -> {ex['detections']} detections")


if __name__ == "__main__":
    main()
