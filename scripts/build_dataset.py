"""Pool the three leakage-free splits into one single-class ``aircraft`` dataset.

Sources (each already audited and split without leakage in its own stream):

    optical  Optical/datasets/airport_disjoint   class plane            -> aircraft
    ir       IRdata/datasets/clean               airplane/drone/heli    -> aircraft
                                                 bird                   -> background
    sar      SAR/datasets/official_scene_clean   class aircraft         -> aircraft

SAR uses SARDet-100K's own split minus val/test images that share a scene with
train, not the re-split `clean` variant: SAR/scripts/overlap_audit.py found that
re-split still leaks overlapping crops of one scene (34.7% of its test images).

Split membership is inherited unchanged, so nothing that was held out in a
stream can appear in training here. Images are symlinked, never copied, and
named ``<prefix>__<original>`` so the sensor is recoverable from the name.

Infrared birds are treated as hard negatives: the frame stays in the dataset
with the bird box removed, so the detector is explicitly taught that a bird is
not an aircraft. Use ``--ir-bird drop-frames`` to discard bird-only frames.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unified_yolo.paths import EXPERTS_ROOT, UNIFIED_ROOT  # noqa: E402

OUT = UNIFIED_ROOT / "datasets" / "unified"

SOURCES = {
    "optical": dict(path=EXPERTS_ROOT / "Optical/datasets/airport_disjoint", prefix="opt", keep=None),
    "ir": dict(path=EXPERTS_ROOT / "IRdata/datasets/clean", prefix="ir", keep={0, 2, 3}),
    "sar": dict(path=EXPERTS_ROOT / "SAR/datasets/official_scene_clean", prefix="sar", keep=None),
}
SPLITS = ("train", "val", "test")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def convert_label(src: Path, keep):
    """Rewrite a YOLO label file as class 0, dropping classes not in ``keep``."""
    lines, dropped = [], 0
    if src.exists():
        for line in src.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            if keep is not None and int(float(parts[0])) not in keep:
                dropped += 1
                continue
            lines.append(" ".join(["0"] + parts[1:]))
    return lines, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ir-bird", choices=["negative", "drop-frames"], default="negative")
    ap.add_argument("--rebuild", action="store_true", help="delete datasets/unified first")
    args = ap.parse_args()

    if args.rebuild and OUT.exists():
        shutil.rmtree(OUT)

    report = {}
    for mod, spec in SOURCES.items():
        src_root: Path = spec["path"]
        if not src_root.exists():
            raise SystemExit(f"missing source split: {src_root}")
        report[mod] = {}
        for split in SPLITS:
            img_dir = OUT / mod / "images" / split
            lbl_dir = OUT / mod / "labels" / split
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)
            n_img = n_box = n_empty = n_dropped_boxes = n_dropped_frames = 0
            for img in sorted((src_root / "images" / split).iterdir()):
                if img.suffix.lower() not in IMG_EXT:
                    continue
                lines, dropped = convert_label(src_root / "labels" / split / (img.stem + ".txt"), spec["keep"])
                if (mod == "ir" and args.ir_bird == "drop-frames" and dropped and not lines):
                    n_dropped_frames += 1
                    continue
                name = f"{spec['prefix']}__{img.name}"
                link = img_dir / name
                if link.is_symlink() or link.exists():
                    link.unlink()
                os.symlink(os.path.realpath(img), link)
                (lbl_dir / f"{spec['prefix']}__{img.stem}.txt").write_text(
                    "\n".join(lines) + ("\n" if lines else ""))
                n_img += 1
                n_box += len(lines)
                n_empty += not lines
                n_dropped_boxes += dropped
            report[mod][split] = dict(images=n_img, boxes=n_box, empty_frames=n_empty,
                                      boxes_dropped=n_dropped_boxes, frames_dropped=n_dropped_frames)

    # clear stale label caches left by earlier builds
    for cache in OUT.rglob("*.cache"):
        cache.unlink()

    # pooled dataset + one yaml per sensor for per-modality evaluation
    def yaml_text(base, rel_dirs):
        lines = [f"path: {base}"]
        for split in SPLITS:
            entries = [f"{d}/images/{split}" for d in rel_dirs]
            lines.append(f"{split}: " + (entries[0] if len(entries) == 1 else "[" + ", ".join(entries) + "]"))
        lines += ["nc: 1", "names: ['aircraft']"]
        return "\n".join(lines) + "\n"

    (OUT / "data.yaml").write_text(
        "# Unified aircraft detection (optical + infrared + SAR) - built by build_dataset.py\n"
        + yaml_text(OUT, list(SOURCES)))
    for mod in SOURCES:
        (OUT / f"data_{mod}.yaml").write_text(yaml_text(OUT / mod, ["."]))

    (OUT / "build_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nwrote {OUT}/data.yaml (+ data_optical/ir/sar.yaml)")


if __name__ == "__main__":
    main()
