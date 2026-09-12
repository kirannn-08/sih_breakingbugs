"""
Train the deadlock-risk predictor.  Run: python3 train_deadlock.py

WHAT IT PREDICTS
----------------
P(this robot is wedged within LABEL_HORIZON seconds), from its own local
observation. Not an action. The architecture doc's own assessment (0.2) is
that full MARL policy control does not belong on the critical path here and
that the defensible learned component is "a small supervised learned
congestion/ETA predictor running onboard, plus an optional EPH-style
deadlock-escape policy behind a rule-based fallback". This is the first of
those two.

The distinction is the whole safety argument. Deadlock RESOLUTION stays
deterministic LIFO: a learned policy that resolves deadlocks sometimes is
worse than a rule that resolves them always. The model only ever adds
caution BEFORE a jam forms, and the rule keeps sole authority for getting out
of one.

SPLIT BY RUN, NOT BY SAMPLE
---------------------------
Samples are taken every 0.5 s, so consecutive samples from one robot are
nearly identical and their labels look 10 s ahead into overlapping futures.
A random split puts near-duplicates of test rows into training, and the
held-out score becomes a measure of memorisation. Both numbers are reported
below precisely because the gap between them is large and worth seeing.

BASELINES ARE REPORTED, ALWAYS
------------------------------
A classifier on a 57% positive dataset can score 57% by answering "yes". Two
reference points are printed with every result: the majority class, and the
hand-written rule the model would have to beat to be worth its inference
cost. A model that does not beat the rule should not be wired in.

THE HEADLINE NUMBER IS THE EARLY-WARNING SLICE, NOT THE OVERALL SCORE
--------------------------------------------------------------------
Overall held-out accuracy is 94.8% and it is NOT the number to quote. Half
the evaluation set is robots that are already stalled, that slice is 99.1%
positive, and `stall_age` ON ITS OWN scores AUC 0.930 -- so most of the
apparent skill is the model reading a stall that has already happened and
restating it. That is a fact about the present, and the whole point of the
model is to be useful before the fact.

The operating regime is a MOVING robot deciding whether to enter a corridor,
so the slice where stall_age is zero is the one that matters. It is harder
and much less flattering: 13.8% positive, AUC 0.839, recall 0.475 at
precision 0.780. Every slice is printed and written to the CSV, with the
stall_age-alone AUC alongside as the leakage reference, so the inflated
number cannot be quoted by accident.

Training on the moving subset alone was tried and scores identically
(AUC 0.839), so the shipped model is trained on everything.
"""

from __future__ import annotations

import csv
import os
import time

import numpy as np

from features import FEATURE_NAMES
from learned import PolicyNet

DATA = "data/deadlock_dataset.npz"
MODEL = "deadlock_net.npz"
REPORT = "results/deadlock_model.csv"
TEST_FRACTION = 0.25


def auc(y: np.ndarray, p: np.ndarray) -> float:
    """ROC AUC by rank (Mann-Whitney U). No sklearn in this project."""
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1)
    # average ranks within ties, or ties bias the statistic
    _, inv, counts = np.unique(p, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2)
                 / (n_pos * n_neg))


def rule_baseline(X: np.ndarray) -> np.ndarray:
    """
    The hand-written predictor the model has to beat.

    "I am in a narrow aisle or at a blind corner, AND a peer in my segment is
    heading towards me." That is the precursor every wedge on this map was
    traced to, written down directly. If a 4k-parameter network cannot beat
    it, the network is not earning its place.
    """
    i = {n: k for k, n in enumerate(FEATURE_NAMES)}
    narrow = (X[:, i["in_narrow"]] > 0.5) | (X[:, i["at_blind_corner"]] > 0.5)
    opposed = np.zeros(len(X), dtype=bool)
    for k in range(3):
        opposed |= ((X[:, i[f"p{k}_opposed"]] > 0.2)
                    & (X[:, i[f"p{k}_same_seg"]] > 0.5))
    return (narrow & opposed).astype(np.float64)


def metrics(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> dict:
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    return {"accuracy": float((pred == y).mean()),
            "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / max(1e-9, prec + rec),
            "auc": auc(y, p)}


def main(seed: int = 0) -> None:
    d = np.load(DATA, allow_pickle=True)
    X, Y, G = d["X"].astype(np.float32), d["Y"].astype(np.int64), d["G"]
    runs = np.unique(G)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(runs)
    n_test = max(1, int(TEST_FRACTION * len(runs)))
    test_runs = set(shuffled[:n_test].tolist())
    te = np.array([g in test_runs for g in G])
    tr = ~te

    # Standardise on TRAINING statistics only, and ship them with the
    # weights: a model normalised with numbers the deployed robot cannot
    # reproduce is a model that silently does something else in the field.
    mu = X[tr].mean(0)
    sd = X[tr].std(0)
    sd[sd < 1e-6] = 1.0
    Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
    Ytr, Yte = Y[tr], Y[te]

    print(f"{len(X)} samples, {len(runs)} runs")
    print(f"  train {len(Xtr):>6} ({Ytr.mean():.1%} positive) "
          f"from {len(runs)-n_test} runs")
    print(f"  test  {len(Xte):>6} ({Yte.mean():.1%} positive) "
          f"from {n_test} held-out runs\n")

    net = PolicyNet(obs_dim=X.shape[1], h1=64, h2=32, n_out=2, seed=seed)
    print(f"model: {X.shape[1]} -> 64 -> 32 -> 2, {net.n_params():,} params")
    t0 = time.time()
    net.train(Xtr, Ytr, epochs=40, lr=0.01, batch=256, verbose=True)
    train_s = time.time() - t0

    p_te = net.forward(Xte)[0][:, 1]
    p_tr = net.forward(Xtr)[0][:, 1]
    m_te, m_tr = metrics(Yte, p_te), metrics(Ytr, p_tr)

    maj = np.full(len(Yte), float(Ytr.mean() > 0.5))
    m_maj = metrics(Yte, maj)
    m_rule = metrics(Yte, rule_baseline(X[te]))

    # The slice that actually matters: a robot still MOVING, i.e. one that can
    # still act on the warning. See the module docstring.
    i_stall = FEATURE_NAMES.index("stall_age")
    moving = X[te][:, i_stall] <= 1e-6
    m_early = metrics(Yte[moving], p_te[moving])
    m_late = metrics(Yte[~moving], p_te[~moving])
    leak_auc = auc(Yte, X[te][:, i_stall])

    t0 = time.time()
    for _ in range(2000):
        net.forward(Xte[:1])
    infer_ms = (time.time() - t0) / 2000 * 1000

    print(f"\ntrained in {train_s:.1f}s   inference {infer_ms:.3f} ms/sample\n")
    rows = [("model_early_warning", m_early), ("model_already_stalled", m_late),
            ("model_heldout_all", m_te), ("model_train_all", m_tr),
            ("baseline_majority", m_maj), ("baseline_rule", m_rule),
            ("leakage_stall_age_alone",
             {"accuracy": float("nan"), "precision": float("nan"),
              "recall": float("nan"), "f1": float("nan"), "auc": leak_auc})]
    hdr = f"{'':<24}{'acc':>8}{'prec':>8}{'rec':>8}{'f1':>8}{'auc':>8}"
    print(hdr)
    print("-" * len(hdr))
    for name, m in rows:
        print(f"{name:<24}{m['accuracy']:>8.3f}{m['precision']:>8.3f}"
              f"{m['recall']:>8.3f}{m['f1']:>8.3f}{m['auc']:>8.3f}")
    print(f"\nHEADLINE (moving robots, n={int(moving.sum())}, "
          f"{Yte[moving].mean():.1%} positive): "
          f"AUC {m_early['auc']:.3f}, recall {m_early['recall']:.3f} "
          f"at precision {m_early['precision']:.3f}")
    print("Do NOT quote the all-slice figure: stall_age alone scores "
          f"AUC {leak_auc:.3f} on it.")

    np.savez(MODEL, W1=net.W1, b1=net.b1, W2=net.W2, b2=net.b2,
             W3=net.W3, b3=net.b3, mu=mu, sd=sd,
             names=np.array(FEATURE_NAMES))
    print(f"\nsaved {MODEL} ({net.n_params():,} params, "
          f"normalisation included)")

    os.makedirs("results", exist_ok=True)
    with open(REPORT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "accuracy", "precision", "recall", "f1", "auc",
                    "n_test", "positive_rate", "params", "inference_ms"])
        for name, m in rows:
            n = (int(moving.sum()) if name == "model_early_warning"
                 else int((~moving).sum()) if name == "model_already_stalled"
                 else len(Yte))
            rate = (float(Yte[moving].mean()) if name == "model_early_warning"
                    else float(Yte[~moving].mean())
                    if name == "model_already_stalled" else float(Yte.mean()))
            w.writerow([name, f"{m['accuracy']:.4f}", f"{m['precision']:.4f}",
                        f"{m['recall']:.4f}", f"{m['f1']:.4f}",
                        f"{m['auc']:.4f}", n, f"{rate:.4f}",
                        net.n_params(), f"{infer_ms:.4f}"])
    print(f"saved {REPORT}")


if __name__ == "__main__":
    main()
