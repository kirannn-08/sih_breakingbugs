"""
Distil WHCA* into the shared local policy.  Run: python3 train.py

Trained on THIS warehouse (`sih_warehouse_grid`), not the generic 40x40 rack
grid it used to use. The generic grid has different aisle widths, no passing
bays and no blind corners, so a policy distilled on it was learning another
building's traffic -- which is exactly the complaint that motivated this
change.

Trained at the SAME density run_study.py evaluates at (tile=1). Training on
a 2x2 tiling and evaluating on the single map cost ~6 points of held-out
agreement and inflated blocked moves tenfold at N=8, which is a train/serve
mismatch and not a property of the policy.
"""
from __future__ import annotations
import time
import numpy as np
from env import LifelongMAPF, sih_warehouse_grid
from policy import MLP, collect

def main(seed=0):
    grid = sih_warehouse_grid(tile=1)
    t0 = time.time()
    print("collecting expert demonstrations (WHCA*, mixed densities) ...")
    Xs, Ys = [], []
    for n in (16, 32, 48):                     # density curriculum
        x, y = collect(grid, n, episodes=2, steps=120, seed=seed + n)
        Xs.append(x); Ys.append(y)
        print(f"  n={n:>2}: {len(x):>6} samples")
    X = np.concatenate(Xs); Y = np.concatenate(Ys)
    print(f"  total {len(X)} samples in {time.time()-t0:.1f}s")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X)); X, Y = X[perm], Y[perm]
    cut = int(0.9 * len(X))
    Xtr, Ytr, Xte, Yte = X[:cut], Y[:cut], X[cut:], Y[cut:]

    net = MLP([X.shape[1], 256, 128, 5], seed=seed)
    print(f"\npolicy: {net.n_params:,} params, obs_dim={X.shape[1]}")
    bs, epochs = 256, 12
    for ep in range(epochs):
        p = rng.permutation(len(Xtr))
        L = A = k = 0
        for i in range(0, len(Xtr) - bs, bs):
            idx = p[i:i + bs]
            l, a = net.train_step(Xtr[idx], Ytr[idx], lr=1e-3)
            L += l; A += a; k += 1
        te = float((net.forward(Xte).argmax(1) == Yte).mean())
        print(f"  epoch {ep+1:>2}  loss {L/k:.4f}  train_acc {A/k:.3f}  held_out {te:.3f}")
    net.save("policy_marl.npz")
    print(f"\nsaved policy_marl.npz  ({net.n_params:,} params)")
    print(f"held-out action agreement with WHCA*: {te:.1%}")
    return net

if __name__ == "__main__":
    main()
