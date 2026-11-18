"""Score a trained unified model separately on each sensor's held-out split.

    python scripts/evaluate.py --name unified_full

Writes outputs/runs/<name>/metrics.json with, per sensor and split: precision,
recall, mAP@0.5, mAP@0.5:0.95 -- plus how often the model's own modality
prediction matches the true sensor (so deployment needs no sensor label), and the
size of each learned FiLM adapter.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from unified_yolo import MODALITIES  # noqa: E402


def modality_accuracy(model, split: str, device: str, per_modality: int = 150, imgsz: int = 640):
    """Fraction of images whose predicted sensor equals the true sensor."""
    import cv2
    import numpy as np
    import torch

    net = model.model.to(device).eval()
    out = {}
    for mod in MODALITIES:
        files = sorted((HERE / "datasets" / "unified" / mod / "images" / split).iterdir())
        step = max(1, len(files) // per_modality)
        files = files[::step][:per_modality]
        hits = 0
        for f in files:
            im = cv2.imread(str(f))
            if im is None:
                continue
            im = cv2.resize(im, (imgsz, imgsz))[:, :, ::-1]
            x = torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1)[None].float().div(255).to(device)
            with torch.no_grad():
                net(x)
            hits += int(net.last_modality_probs[0].argmax()) == MODALITIES.index(mod)
        out[mod] = round(hits / max(len(files), 1), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--weights", default=None, help="default: outputs/runs/<name>/weights/best.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--conf", type=float, default=0.001)
    args = ap.parse_args()

    from ultralytics import YOLO

    run_dir = HERE / "outputs" / "runs" / args.name
    weights = args.weights or str(run_dir / "weights" / "best.pt")
    model = YOLO(weights)

    results = {"run": args.name, "weights": weights}
    for mod in MODALITIES:
        data = str(HERE / "datasets" / "unified" / f"data_{mod}.yaml")
        results[mod] = {}
        for split in ("val", "test"):
            res = model.val(data=data, split=split, imgsz=args.imgsz, batch=args.batch,
                            device=args.device, conf=args.conf, plots=False, verbose=False,
                            project=str(run_dir), name=f"eval_{mod}_{split}", exist_ok=True)
            results[mod][split] = {
                "mAP50": round(float(res.box.map50), 4),
                "mAP50_95": round(float(res.box.map), 4),
                "precision": round(float(res.box.mp), 4),
                "recall": round(float(res.box.mr), 4),
            }
            print(mod, split, results[mod][split])

    inner = model.model
    # the "off" arm never runs QualityNet, so it has no sensor prediction to score
    if hasattr(inner, "film_report") and getattr(inner, "cond_mode", "off") != "off":
        results["film_weight_magnitude"] = {str(k): round(v, 5) for k, v in inner.film_report().items()}
        results["modality_prediction_accuracy_test"] = modality_accuracy(model, "test", args.device)

    out = run_dir / "metrics.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nmetrics -> {out}")


if __name__ == "__main__":
    main()
