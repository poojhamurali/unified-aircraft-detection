"""Route, fall back, or refuse: does a sensor router help aircraft detection?

Every image of the three audited test splits is scored six ways, each sending it to
a detector chosen differently:

    oracle    the expert of its true sensor (the best any router can do)
    stats     the expert picked by a random forest on the 8 image statistics
    cnn       the expert picked by the tiny thumbnail CNN
    vote      the expert of the averaged stats + CNN probabilities
    fallback  vote, except that images on which the two routers disagree, or whose
              averaged confidence is below 0.8, go to the pooled detector
    pooled    the pooled YOLOv8s (unified_off) for every image

A refuse rule rejects images whose statistics are unlikely under every sensor's
training distribution (one Gaussian per sensor; threshold = 1st percentile of the
validation images). It is scored on the audited test sets, where every refusal is a
mistake, and on 48 aircraft-free MAN tiles (~10 m/pixel optical), where the optical
expert fires false aircraft.

Leave-one-satellite-out: the routers are retrained without one SAR source, and that
source's images are checked for whether they still reach the SAR expert.

Test-set predictions are captured inside Ultralytics' validator, not from predict():
on the same image the two return different boxes, and only the validator's reproduce
the reported test mAP. Each detector is validated once per sensor, its per-image
true-positive matrix is cached, and each arm then picks per image whose matrix to use,
so the oracle arm equals the experts' val() numbers by construction. The IR expert's
bird predictions are dropped before matching. MAN tiles have no labels and are scored
from predict(), counting only confident (>= 0.25) detections.

    ../.venv/bin/python scripts/route_detect.py    # outside the sandbox (MPS)
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
sys.path.insert(0, str(HERE / "scripts"))
from router_improvements import TinyCNN, cnn_proba, to_tensor, train_cnn  # noqa: E402,F401
from unified_yolo import MODALITIES, image_statistics  # noqa: E402
from unified_yolo.paths import EXPERTS_ROOT as ROOT, MAN_TILES_DIR  # noqa: E402

OUT = HERE / "outputs" / "route_detect"
DATA = HERE / "datasets" / "unified"
EXPERTS = {
    "optical": (ROOT / "Optical/outputs/runs/airport_disjoint_yolov8s_norot/weights/best.pt", 640, None),
    "ir": (ROOT / "IRdata/outputs/runs/top2_yolo11s/weights/best.pt", 640, [0, 2, 3]),
    "sar": (ROOT / "SAR/outputs/runs/official_yolov8s/weights/best.pt", 512, None),
}
POOLED = (HERE / "outputs/runs/unified_off/weights/best.pt", 640, None)
MAN_TILES = sorted(MAN_TILES_DIR.glob("*_raw.png"))
ROUTE_CONF = 0.8
DETECT_CONF = 0.25
SAR_SOURCE = {512: "Gaofen-3 (SAR-AIRcraft)", 224: "TerraSAR-X (SADD)", 256: "HISEA-1 (MSAR)"}
GF3_TRAIN_SAMPLE = 1500


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #
def featurize(paths):
    stats, thumbs, sizes = [], [], []
    for p in paths:
        im0 = cv2.imread(str(p))
        sizes.append(im0.shape[0])
        im = cv2.resize(im0, (640, 640), interpolation=cv2.INTER_AREA)[:, :, ::-1].copy()
        stats.append(image_statistics(to_tensor(im))[0].numpy())
        thumbs.append(cv2.resize(im, (128, 128), interpolation=cv2.INTER_AREA))
    return np.stack(stats), np.stack(thumbs), np.array(sizes)


def split_paths(split):
    out = []
    for m, mod in enumerate(MODALITIES):
        imgs = sorted((DATA / mod / "images" / split).iterdir())
        if split == "train" and mod == "sar":
            sizes = {p: cv2.imread(str(p)).shape[0] for p in imgs}
            small = [p for p in imgs if sizes[p] != 512]
            gf3 = [p for p in imgs if sizes[p] == 512]
            rng = np.random.default_rng(0)
            gf3 = [gf3[i] for i in sorted(rng.choice(len(gf3), GF3_TRAIN_SAMPLE, replace=False))]
            imgs = small + gf3
        out += [(p, m) for p in imgs]
    return out


def load_features():
    cache = OUT / "features.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    feats = {}
    for split in ("train", "val", "test"):
        items = split_paths(split)
        X, T, S = featurize([p for p, _ in items])
        feats[split] = {"paths": [str(p) for p, _ in items], "y": np.array([m for _, m in items]),
                        "X": X, "T": T, "size": S}
        print(f"  features {split}: {len(items)} images")
    X, T, S = featurize(MAN_TILES)
    feats["man"] = {"paths": [str(p) for p in MAN_TILES], "X": X, "T": T, "size": S}
    cache.write_bytes(pickle.dumps(feats))
    return feats


# --------------------------------------------------------------------------- #
# routers and the refuse rule
# --------------------------------------------------------------------------- #
def fit_routers(X, T, y):
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(400, random_state=0, class_weight="balanced").fit(X, y)
    torch.manual_seed(0)
    net = train_cnn(T, y)
    return rf, net


def route_probs(rf, net, X, T):
    p_rf = rf.predict_proba(X)
    p_cnn = cnn_proba(net, T)
    return p_rf, p_cnn, (p_rf + p_cnn) / 2


class SensorGaussians:
    """One Gaussian per sensor over the 8 statistics; score = best log-likelihood."""

    def fit(self, X, y, reg=1e-3):
        self.params = []
        for m in range(len(MODALITIES)):
            Xm = X[y == m]
            cov = np.cov(Xm.T) + reg * np.eye(X.shape[1])
            self.params.append((Xm.mean(0), np.linalg.inv(cov), np.linalg.slogdet(cov)[1]))
        return self

    def score(self, X):
        out = []
        for mu, icov, logdet in self.params:
            d = X - mu
            out.append(-0.5 * (np.einsum("ij,jk,ik->i", d, icov, d) + logdet + X.shape[1] * np.log(2 * np.pi)))
        return np.max(out, axis=0)


# --------------------------------------------------------------------------- #
# detection
# --------------------------------------------------------------------------- #
def capture(name, spec):
    """Validate one detector on every sensor's test split; per image (tp, conf, n_gt)."""
    cache = OUT / f"val_{name}.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionValidator

    weights, imgsz, classes = spec
    records = {}
    original = DetectionValidator.update_metrics

    def hook(self, preds, batch):
        for si, pred in enumerate(preds):
            pb = self._prepare_batch(si, batch)
            boxes = pred["bboxes"].cpu().numpy().astype(np.float32)
            conf = pred["conf"].cpu().numpy().astype(np.float32)
            if classes is not None:
                keep = np.isin(pred["cls"].cpu().numpy().astype(int), classes)
                boxes, conf = boxes[keep], conf[keep]
            gt = pb["bboxes"].cpu().numpy().astype(np.float32).reshape(-1, 4)
            records[Path(batch["im_file"][si]).name] = (match(boxes, gt), conf, len(gt))
        return original(self, preds, batch)

    DetectionValidator.update_metrics = hook
    try:
        model = YOLO(str(weights))
        for mod in MODALITIES:
            model.val(data=str(DATA / f"data_{mod}.yaml"), split="test", imgsz=imgsz, batch=8,
                      device="mps", conf=0.001, iou=0.7, agnostic_nms=classes is not None,
                      plots=False, verbose=False, project=str(OUT / "val"),
                      name=f"{name}_{mod}", exist_ok=True)
    finally:
        DetectionValidator.update_metrics = original
    cache.write_bytes(pickle.dumps(records))
    print(f"  validated {name}: {len(records)} images")
    return records


def predict_all(name, spec, paths):
    cache = OUT / f"preds_{name}.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    from ultralytics import YOLO

    weights, imgsz, classes = spec
    model = YOLO(str(weights))
    preds = {}
    for p in paths:
        r = model.predict(p, imgsz=imgsz, conf=0.001, iou=0.7, max_det=300, device="mps",
                          classes=classes, agnostic_nms=classes is not None, verbose=False)[0]
        preds[p] = (r.boxes.xyxy.cpu().numpy().astype(np.float32),
                    r.boxes.conf.cpu().numpy().astype(np.float32), r.orig_shape)
    cache.write_bytes(pickle.dumps(preds))
    print(f"  predicted {name}: {len(paths)} images")
    return preds


IOUV = np.linspace(0.5, 0.95, 10)


def match(pred, gt):
    """Ultralytics' match_predictions for a single class."""
    from ultralytics.utils.metrics import box_iou

    correct = np.zeros((len(pred), len(IOUV)), dtype=bool)
    if not len(pred) or not len(gt):
        return correct
    iou = box_iou(torch.from_numpy(gt), torch.from_numpy(pred)).numpy()
    for i, t in enumerate(IOUV):
        m = np.array(np.nonzero(iou >= t)).T
        if m.shape[0]:
            if m.shape[0] > 1:
                m = m[iou[m[:, 0], m[:, 1]].argsort()[::-1]]
                m = m[np.unique(m[:, 1], return_index=True)[1]]
                m = m[np.unique(m[:, 0], return_index=True)[1]]
            correct[m[:, 1].astype(int), i] = True
    return correct


def mean_ap(chosen, names):
    from ultralytics.utils.metrics import ap_per_class

    tps, confs, n_gt = [], [], 0
    for n in names:
        tp, conf, ng = chosen(n)
        tps.append(tp)
        confs.append(conf)
        n_gt += ng
    tp = np.concatenate(tps) if tps else np.zeros((0, 10), bool)
    conf = np.concatenate(confs) if confs else np.zeros(0)
    ap = ap_per_class(tp, conf, np.zeros(len(conf)), np.zeros(n_gt))[5]
    return round(float(ap[:, 0].mean()), 4), round(float(ap.mean()), 4)


# --------------------------------------------------------------------------- #
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    feats = load_features()
    tr, va, te, man = feats["train"], feats["val"], feats["test"], feats["man"]

    rf, net = fit_routers(tr["X"], tr["T"], tr["y"])
    p_rf, p_cnn, p_vote = route_probs(rf, net, te["X"], te["T"])
    gauss = SensorGaussians().fit(tr["X"], tr["y"])
    tau = float(np.percentile(gauss.score(va["X"]), 1))
    refuse_te = gauss.score(te["X"]) < tau

    names = [Path(p).name for p in te["paths"]]
    recs = {mod: capture(mod, EXPERTS[mod]) for mod in MODALITIES}
    recs["pooled"] = capture("pooled", POOLED)
    preds = {mod: predict_all(mod, EXPERTS[mod], man["paths"]) for mod in MODALITIES}
    preds["pooled"] = predict_all("pooled", POOLED, man["paths"])

    agree = p_rf.argmax(1) == p_cnn.argmax(1)
    confident = agree & (p_vote.max(1) >= ROUTE_CONF)
    choice = {
        "oracle": [MODALITIES[m] for m in te["y"]],
        "stats": [MODALITIES[m] for m in p_rf.argmax(1)],
        "cnn": [MODALITIES[m] for m in p_cnn.argmax(1)],
        "vote": [MODALITIES[m] for m in p_vote.argmax(1)],
        "fallback": [MODALITIES[m] if ok else "pooled" for m, ok in zip(p_vote.argmax(1), confident)],
        "pooled": ["pooled"] * len(names),
    }
    index = {n: i for i, n in enumerate(names)}

    results = {"route_conf": ROUTE_CONF, "refuse_threshold_loglik": round(tau, 3), "arms": {}}
    for arm, picks in list(choice.items()) + [("vote+refuse", choice["vote"])]:
        def chosen(n, picks=picks, arm=arm):
            i = index[n]
            if arm == "vote+refuse" and refuse_te[i]:
                return np.zeros((0, len(IOUV)), bool), np.zeros(0, np.float32), recs["pooled"][n][2]
            return recs[picks[i]][n]
        row = {}
        for m, mod in enumerate(MODALITIES):
            sel = [n for n, y in zip(names, te["y"]) if y == m]
            ids = np.array([index[n] for n in sel])
            m50, m5095 = mean_ap(chosen, sel)
            row[mod] = {"images": len(sel), "mAP50": m50, "mAP50_95": m5095,
                        "to_own_expert": round(float(np.mean([picks[i] == mod for i in ids])), 4),
                        "to_pooled": round(float(np.mean([picks[i] == "pooled" for i in ids])), 4)}
            if arm == "vote+refuse":
                row[mod]["refused"] = int(refuse_te[ids].sum())
        results["arms"][arm] = row
        print(arm, {k: (v["mAP50"], v["mAP50_95"], v["to_own_expert"]) for k, v in row.items()})

    # aircraft-free MAN tiles: false aircraft at conf >= DETECT_CONF
    mr, mc, mv = route_probs(rf, net, man["X"], man["T"])
    man_refused = gauss.score(man["X"]) < tau
    routed = [MODALITIES[m] for m in mv.argmax(1)]

    def false_aircraft(model_of):
        return int(sum((preds[model_of(i, p)][p][1] >= DETECT_CONF).sum()
                       for i, p in enumerate(man["paths"]) if model_of(i, p)))

    results["man_tiles"] = {
        "tiles": len(man["paths"]),
        "routed_to": {mod: routed.count(mod) for mod in MODALITIES},
        "refused": int(man_refused.sum()),
        "false_aircraft_optical_expert": false_aircraft(lambda i, p: "optical"),
        "false_aircraft_routed_expert": false_aircraft(lambda i, p: routed[i]),
        "false_aircraft_pooled": false_aircraft(lambda i, p: "pooled"),
        "false_aircraft_routed_with_refuse": false_aircraft(
            lambda i, p: None if man_refused[i] else routed[i]),
        "loglik_man_median": round(float(np.median(gauss.score(man["X"]))), 3),
        "loglik_test_median": round(float(np.median(gauss.score(te["X"]))), 3),
    }
    print("MAN:", results["man_tiles"])

    # leave one SAR satellite out of router training
    loso = {}
    sar = MODALITIES.index("sar")
    for size, source in SAR_SOURCE.items():
        keep = ~((tr["y"] == sar) & (tr["size"] == size))
        rf_l, net_l = fit_routers(tr["X"][keep], tr["T"][keep], tr["y"][keep])
        g_l = SensorGaussians().fit(tr["X"][keep], tr["y"][keep])
        tau_l = float(np.percentile(g_l.score(va["X"][(va["y"] != sar) | (va["size"] != size)]), 1))
        # every image of the held-out source, train split included: none were used above
        parts = [(d, (d["y"] == sar) & (d["size"] == size)) for d in (tr, va, te)]
        Xh = np.concatenate([d["X"][m] for d, m in parts])
        Th = np.concatenate([d["T"][m] for d, m in parts])
        a, b, v = route_probs(rf_l, net_l, Xh, Th)
        loso[source] = {
            "held_out_images": int(len(Xh)),
            "routed_to_sar": {"stats": round(float((a.argmax(1) == sar).mean()), 4),
                              "cnn": round(float((b.argmax(1) == sar).mean()), 4),
                              "vote": round(float((v.argmax(1) == sar).mean()), 4)},
            "refused_as_unusual": round(float((g_l.score(Xh) < tau_l).mean()), 4),
        }
        print("LOSO", source, loso[source])
    results["leave_one_sar_satellite_out"] = loso

    (OUT / "route_detect.json").write_text(json.dumps(results, indent=2))
    print(f"\n-> {OUT / 'route_detect.json'}")


if __name__ == "__main__":
    main()
