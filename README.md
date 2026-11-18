# Unified Aircraft Detection Across Optical, Thermal-Infrared and SAR Imagery

One YOLOv8s detector for all three sensors, trained on leakage-audited splits, with a
lightweight sensor router that refuses imagery no trained sensor supports. Part of the
M.Tech project *Leakage-Audited Aircraft Detection Across Optical, Thermal-Infrared
and SAR Imagery* (Poojha Muralidharan, Amrita School of Computing).

**Project website:** https://poojhamurali.github.io/unified-aircraft-detection/

The three sensor-specific experts and the dataset audits live in the companion
repository [aircraft-detection-yolo-experts](https://github.com/poojhamurali/aircraft-detection-yolo-experts).

## System

```
any image (optical, thermal or SAR)
   -> router: 8 image statistics + small CNN; unlike every trained sensor? -> refuse
   -> one pooled YOLOv8s -> aircraft boxes
```

No sensor label is needed at test time.

## Results (test mAP@0.5 / mAP@0.5:0.95)

| Model | Optical | Thermal IR | SAR |
|---|---|---|---|
| Sensor expert (one model per sensor) | 0.993 / 0.859 | 0.688 / 0.382 | 0.783 / 0.450 |
| **One pooled YOLOv8s (recommended)** | **0.993 / 0.855** | **0.704 / 0.380** | **0.809 / 0.437** |
| + sensor token (FiLM) | 0.993 / 0.855 | 0.721 / 0.385 | 0.788 / 0.422 |
| + image statistics (FiLM) | 0.993 / 0.856 | 0.714 / 0.386 | 0.796 / 0.425 |
| + both (FiLM) | 0.993 / 0.855 | 0.711 / 0.372 | 0.779 / 0.414 |

* The pooled detector matches the experts on all three sensors; FiLM conditioning on
  a sensor token or image statistics adds no consistent gain.
* The router sends 99.5% (statistics), 99.97% (CNN) and 100% (vote) of test images to
  the right expert, yet routing to experts does not beat the pooled detector.
* The refusal rule rejects all 48 aircraft-free out-of-domain tiles, on which the
  routed experts fire 614 false aircraft, at the cost of 1.5% of test images.

Thermal "aircraft" are airplanes, drones and helicopters; birds are background.

## How the single model is trained

* One class, `aircraft`, over the union of the three audited training sets.
* Each epoch draws 2,000 images per sensor, so the larger SAR set cannot dominate.
* Mosaic augmentation only tiles images of the same sensor.
* Augmentation valid for all three sensors: no hue jitter, rotation or vertical flip.
* FiLM variants (`unified_yolo/model.py`) start as exactly the pretrained YOLOv8s
  (zero-initialised adapters), checked by `scripts/smoke_test.py`.

## Setup

Clone both repositories side by side and prepare the expert datasets first (see the
experts repository):

```
workspace/
  aircraft-detection-yolo-experts/   Optical/ IRdata/ SAR/
  unified-aircraft-detection/        this repository
```

Another location can be given with `AIRCRAFT_EXPERTS_ROOT=/path/to/experts`.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/build_dataset.py --rebuild     # pool the three audited splits
python scripts/smoke_test.py                  # wiring checks on CPU, ~10 s
bash scripts/run_ablation.sh 30               # off / modality_only / quality_only / full
python scripts/route_detect.py                # routing, fallback and refusal evaluation
```

Training was run on an Apple-silicon GPU (PyTorch MPS backend) and must run outside
any sandbox that hides the GPU.

## Layout

```
unified_yolo/model.py      QualityNet, FiLM layers, UnifiedDetectionModel
unified_yolo/trainer.py    balanced sensor sampler, same-sensor mosaic, trainer
unified_yolo/paths.py      where the experts' data and weights are found
scripts/build_dataset.py   pooled single-class dataset from the three audited splits
scripts/train.py           one training arm (off, modality_only, quality_only, full)
scripts/evaluate.py        per-sensor validation/test metrics
scripts/stat_router.py     statistics router (first round)
scripts/router_*.py        router comparison, robustness and full-data fit
scripts/route_detect.py    detection under each routing strategy, refusal, unseen satellites
```
