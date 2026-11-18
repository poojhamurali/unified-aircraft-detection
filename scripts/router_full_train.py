"""Refit the statistics router on *all* optical/IR training images (+1,500 random SAR).

The first router used an evenly spaced 400-image sample per sensor. For infrared
that sample contained none of the 80 colour-encoded training frames, so the
router learned "colour means optical" and sent coloured IR test frames to the
optical expert. This script checks whether full coverage removes that error.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
from router_improvements import read, summarize, to_tensor  # noqa: E402
from unified_yolo import MODALITIES, image_statistics  # noqa: E402

OUT = HERE / "outputs" / "stat_router"


def items(split, cap_sar=None, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for m, mod in enumerate(MODALITIES):
        imgs = sorted((HERE / "datasets/unified" / mod / "images" / split).iterdir())
        if mod == "sar" and cap_sar and len(imgs) > cap_sar:
            imgs = [imgs[i] for i in sorted(rng.choice(len(imgs), cap_sar, replace=False))]
        out += [(p, m) for p in imgs]
    return out


def stats(items_):
    X = np.stack([image_statistics(to_tensor(read(p)))[0].numpy() for p, _ in items_])
    return X, np.array([m for _, m in items_])


def main():
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.ensemble import RandomForestClassifier

    tr = items("train", cap_sar=1500)
    te = items("test", cap_sar=1000, seed=1)
    print(f"train {len(tr)} / test {len(te)} images")
    Xtr, ytr = stats(tr)
    Xte, yte = stats(te)
    np.savez(OUT / "stats_full.npz", Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)

    res = {}
    for name, clf in (("gaussian", QuadraticDiscriminantAnalysis(reg_param=0.05)),
                      ("random_forest", RandomForestClassifier(400, random_state=0,
                                                               class_weight="balanced"))):
        clf.fit(Xtr, ytr)
        res[name] = summarize(yte, clf.predict(Xte))
        print(name, res[name])
    (OUT / "router_full_train.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
