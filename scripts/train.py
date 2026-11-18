"""Train the unified optical + infrared + SAR aircraft detector.

    python scripts/train.py --name unified_full --mode full --epochs 30

``--mode`` selects the ablation arm (all share data, sampler, seed, schedule):
    full           quality vector + modality token + FiLM  (proposed)
    modality_only  modality token + FiLM
    quality_only   quality vector + FiLM
    off            plain pooled YOLO (no conditioning)      (baseline)

Must run outside a sandbox on Apple silicon (Metal is invisible inside one).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from unified_yolo.paths import pretrained_weights  # noqa: E402

# Hyper-parameters that suit all three sensors at once: no hue (two of three are
# single-band), no vertical flip (sky scenes), gentle scale so few-pixel infrared
# targets survive, no mixup/copy-paste (they destroy low-contrast targets).
HYP = dict(
    hsv_h=0.0, hsv_s=0.3, hsv_v=0.3,
    degrees=0.0, translate=0.1, scale=0.3, shear=0.0, perspective=0.0,
    flipud=0.0, fliplr=0.5,
    mosaic=1.0, close_mosaic=3, mixup=0.0, copy_paste=0.0,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--mode", default="full", choices=["full", "modality_only", "quality_only", "off"])
    ap.add_argument("--weights", default=pretrained_weights(),
                    help="pretrained YOLO checkpoint to transfer into the unified detector")
    ap.add_argument("--data", default=str(HERE / "datasets" / "unified" / "data.yaml"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--epoch-size", type=int, default=6000,
                    help="images drawn per epoch (balanced across sensors)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=4,
                    help="4 keeps the dense optical/SAR scenes inside memory with the MPS assigner patch")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cache", default="False", choices=["False", "ram", "disk"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-balance", action="store_true", help="disable equal-per-sensor sampling")
    ap.add_argument("--film-layers", default="2,4,6,9,12,15,18,21")
    ap.add_argument("--aux-weight", type=float, default=0.1)
    ap.add_argument("--fraction", type=float, default=1.0, help="debug: use a fraction of the train set")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    import torch
    if args.device == "mps" and not torch.backends.mps.is_available() and not args.allow_cpu:
        raise SystemExit("MPS unavailable (sandboxed?). Run outside the sandbox or pass --allow-cpu.")

    run_dir = HERE / "outputs" / "runs" / args.name
    if (run_dir / "results.csv").exists() and not args.force:
        raise SystemExit(f"{run_dir} already holds a run; delete it or pass --force.")

    import mps_tal_patch
    if mps_tal_patch.apply(args.device):
        print("[patch] MPS TaskAlignedAssigner workaround installed")

    from unified_yolo.trainer import UnifiedTrainer

    UnifiedTrainer.model_kwargs = dict(
        film_layers=tuple(int(i) for i in args.film_layers.split(",")),
        cond_mode=args.mode, aux_weight=args.aux_weight)
    UnifiedTrainer.epoch_size = args.epoch_size
    UnifiedTrainer.balanced = not args.no_balance

    overrides = dict(
        model=args.weights, data=args.data, epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch, device=args.device, workers=args.workers,
        cache=False if args.cache == "False" else args.cache,
        project=str(HERE / "outputs" / "runs"), name=args.name, exist_ok=True,
        seed=args.seed, deterministic=True, plots=True, val=True,
        fraction=args.fraction, single_cls=False, **HYP)
    trainer = UnifiedTrainer(overrides=overrides)
    trainer.train()
    print(f"\nbest weights: {trainer.best}")


if __name__ == "__main__":
    main()
