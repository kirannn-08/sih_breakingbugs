# CLAUDE.md — project context

Read this before touching anything. Then read `docs/ARCHITECTURE.md` (design
intent) and `docs/ENGINEERING_LOG.md` (bugs already hit — do not reintroduce
them).

---

## What this is

Decentralized fleet coordination for warehouse AMRs. Smart India Hackathon
PS 26123. Python simulation, intended to transfer to ROS 2 + Gazebo and then
to hardware.

**The central claim, which everything serves:** the operator says *what*
needs doing; the robots decide *who* and *how*. There is no central planner.
A server broadcasts tasks and collects telemetry — it never commands.

## Current state

Numbers below are 4 robots, 180 s, 20 seeds, **with occluded sensing**. Every
earlier figure in this file was measured with a sensor that saw through solid
racking; they are not comparable and have been replaced.
See `docs/DEADLOCK_RESOLUTION.md` for the full write-up.

- **56/57 tests passing. `test_T18_beats_stop_and_wait` FAILS at +15.9%**
  (target 20%). Left failing deliberately — the case that it measures the
  wrong quantity is in DEADLOCK_RESOLUTION.md §7.1, not in a weakened assert.
- **Two headline metrics, and they disagree.** `avg_task_time` averages over
  *completed* tasks only, so finishing the hard tasks the baseline abandons
  drags the mean up. Seed 3: baseline 1 task, full system 5. **Throughput is
  the unbiased one: 4.7 → 6.0 tasks/run, +30.1%, paired t = 2.55.** Quote
  throughput; quote task time only alongside it.
- Default map, B3 full: **+15.9%** task time, **+30.1%** throughput,
  **0 collisions** across all 140 runs of all 7 arms.
- **LiDAR is occluded** (`WarehouseMap.line_of_sight`, Amanatides-Woo) and now
  drives *decisions*, not just the brake: `perception.LidarTracker`,
  `lidar_scan` (comms-free obstacle marking), `lidar_speed_cap`, and a
  blind-corner rule at aisle mouths. Turning occlusion on alone produced 182
  collisions on seed 1 — the brake had been relying on omniscience.
- **Deadlock resolution is LIFO**, not priority: the last robot into the
  contested region backs out along its own trail, and the stack includes
  everyone *queued behind* the cycle, not just cycle members. Recovery went
  from *never* (a 2-cycle held 4589 robot-ticks on seed 3) to 1.4–20 s.
  84 backouts started / 82 completed over 20 seeds.
- **Space-time A\* is fixed** (flaw 2): state is `(x, y, t_bucket)` with a
  WAIT action; `plan` keeps its adjacency contract, `plan_st`/`schedule_of`
  carry the timing, and a planned wait executes as a computed slowdown.
  Confirmed live: 215 WAIT actions across 147 plans. Latency 16.2 ms.
- **B1 (congestion routing alone) is still worth nothing: −0.7%, t = 0.12** —
  even after the space-time fix. The earlier diagnosis that flaw 2 explained
  it was incomplete.
- **The LIFO stack's throughput gain is not statistically established**
  (+0.60 tasks, t = 1.75, n = 20). It demonstrably resolves jams that never
  resolved before; that it raises throughput is suggested, not proven.
- **Learned deadlock-risk model**: trained on THIS map from 58,763 synthetic
  samples, 6,178 params, 0.007 ms. Held-out **AUC 0.839 on moving robots** —
  *not* the 0.975 all-slice figure, which is inflated (`stall_age` alone
  scores 0.930 on it). **In the loop it does not beat the deterministic rule**
  (t = 1.45), so it ships **off by default**. Do not quote it as a win.
- **The MARL policy loses to a classical planner** on our own map: 87.1%
  held-out action agreement, but 3–33% of WHCA\* throughput, beaten by
  greedy+jitter at every N ≥ 16. Compounding covariate shift. EPH is cited
  prior art, **not implemented**.
- `use_policy` is **still dead** (DEAD_CODE.md Tier 3). `use_risk_model` is
  the one that is consumed.
- **amr_msgs is 1.1.0.** `WaitFor` gained `entered_at`, `segment_id`,
  `dist_to_block`, `backing_out` — additive only, all defaulted, 1.0.0
  receivers unaffected. Flagged per the frozen-contract rule.
- Collisions/near-misses now count **events, not ticks** (BUG 7's defect,
  left in this metric). One 136 s graze had reported as 1359 collisions.
- Dashboard: `python3 dashboard_server.py --port 8080` (needs tornado)
- SIH layout: **`run_sih_layout.py` is still broken** — `Simulation(wmap=...,
  starts=...)` matches no signature. **The +20.0% SIH figure remains
  unreproducible — do not quote it.**
- Only runtime dependency is numpy

## Commands

```bash
python3 -m pytest tests/ -q      # 57 tests, ~35 s
python3 run_benchmark.py 10      # headline number
python3 run_sih_layout.py        # the SIH drawing's layout
python3 demo_scenarios.py        # 7 demo scenarios
python3 train_policy.py          # retrain action policy (~4 s)
python3 synth_data.py            # regenerate synthetic deadlock data (~5 min)
python3 train_deadlock.py        # train the deadlock-risk model (~1 s)
cd rl && python3 train.py && python3 run_study.py   # MAPF scaling on our map
```

---

## THE WORK: five known flaws, in priority order

A code review found five real defects. All confirmed. Fix in this order,
**one commit each**, running the test suite after every change.

### 1. The auction is a central auctioneer — ✅ FIXED (see ENGINEERING_LOG BUG 8)

`sim2d.py :: Simulation.run_auction` collects bids by reaching into every
robot object directly and resolves the winner centrally. **No `Bid` message
is ever sent.** `MsgType.BID` appears only in the receive handler;
`Robot.pending_bids` is populated but never read — dead code.

This contradicts the project's core claim, and it's findable by reading one
method.

**Fixed as specified.** Bidding lives in `Robot.submit_bid` /
`Robot.step_auction` / `Robot._on_claim`; `Simulation.run_auction` is deleted.
Timings `T_BID` / `T_CLAIM` / `T_REBID` are in `coordination.py`.
Verified live, not a no-op: 1632 `BID` + 13 `CLAIM` per 180 s run, and
`RELEASE` fires 0× lossless / 1× at 20% loss — the double-claim branch
triggers only where it should. B4 went **+24.8% → +16.2%**, exactly the
predicted direction.

### 2. Space-time A\* has no time in its state — ✅ FIXED

`planner.py :: plan` — the closed set is `set[tuple[int, int]]`. Spatial only.

Time affects the *cost* (`t_arrive` drives the occupancy lookup) but not the
*state*, so the first expansion of a cell wins permanently. The planner can
never discover "wait here 5 s for traffic to clear." This is a correctness
bug, not a missing feature: the cost function is time-dependent while the
search assumes it isn't.

**Fix:** state becomes `(x, y, t_bucket)` with time discretized (~1 s).
Add an explicit WAIT action. Keep `heuristic()` admissible — it must stay
distance/v_max, with **no congestion term**, or A\* stops being correct.

### 3. Instant heading snap (no differential-drive kinematics)

`sim2d.py :: Robot.step` does `self.theta = math.atan2(...)` — the robot
rotates instantaneously.

Real spec: 1.00 m × 0.98 m (wheels included), wheel separation 0.86 m. Real
AMRs stop, rotate, then drive. Because rotation is free here, broadcast ETAs
are optimistic, and ETA error is what desynchronizes conflict prediction on
real hardware.

**Fix:** add `max_omega`, rotate toward the heading at a bounded rate, and
penalize turns in the planner cost so ETA and reality agree.

### 4. Omniscient LiDAR (sees through racks) — ✅ FIXED

`sim2d.py :: Simulation.step` builds `sensed` from pure Euclidean distance
with no occlusion check. Robots detect peers through solid racks and around
corners, so braking is better than reality.

**Fix:** ray-cast against the static grid before adding a peer to `sensed`.

### 5. `orca_adjust` is not ORCA

`sim2d.py :: Robot.orca_adjust` is an artificial potential field with a
hardcoded "dodge right" rule (`rx, ry = hy, -hx`). Real ORCA solves an LP in
velocity-obstacle space.

**Fix:** rename it (`reciprocal_sidestep`) and update comments, README and
`docs/SCOPE.md` so nothing claims ORCA. Real ORCA arrives with ROS 2 via
Python-RVO2 or Nav2's DWB/TEB with footprint padding — do not write it from
scratch here.

---

## GUARDRAILS — read before you start

**Expect the numbers to get worse, and do not fight it.** This happened
exactly as predicted when #1 landed: `B4_full_20pct_loss` fell from +24.8% to
**+16.2%**, and B3 from +23.2% to **+22.1%**. Median allocation latency went
from in-process to 0.40 s lossless / 1.10 s at 20% loss. **A drop is the fix
working.** Never revert a correctness fix to recover a benchmark number —
report the honest figure instead.

**Every past shortcut biased results favourably.** Omniscient sensing,
instant turning, central auction, no time dimension — not one made the system
look worse. That is the signature of fitting to the scoring criteria rather
than to physics. When a change makes a number go up, check whether it did so
by improving the system or by weakening the test.

**A guard that never fires is worse than no guard.** Two bugs in
`ENGINEERING_LOG.md` (#4, #5) were silent no-ops that looked like working
code. **Byte-identical benchmark output after a behavioural change means the
code path is dead.** Instrument every new rule and confirm it triggers.

**Do not weaken the tests to make them pass.** In particular
`test_tasks_actually_complete` blocks the degenerate "park everything, zero
collisions" solution, and `test_deterministic_given_seed` is what makes the
benchmark reproducible. If a test fails after a fix, the fix or the
expectation is wrong — not the test.

**Ask before changing calibration constants.** `r_hard` / `r_slow` in
`sim2d.py` are tuned so the default radius reproduces the original 0.50/1.05
margins. An earlier refactor moved them by 0.04 m and silently cost 6
percentage points on the headline result.

---

## Invariants that must not break

1. **L2 safety supervisor has final authority over speed and can only ever
   reduce it.** No learned component sits below L4.
2. **Safety runs on LiDAR, not comms** — that is why a total blackout
   produces zero collisions.
3. **All peer traffic goes through `comms.py`.** No robot may read another
   robot's state directly. Breaking this silently invalidates the entire
   decentralization claim. (#1 above is exactly this violation.)
4. **Stopping is a computed last resort**, reached only when the required
   speed falls below `v_min` — never a reflex. That mechanism is where the
   throughput gain comes from.
5. `amr_msgs.py` is a **frozen contract**. Changing a field breaks the
   dashboard and comms workstreams — flag it, don't just edit it.

## Unverified assumptions (SIH layout)

`warehouse_map_sih.py` header documents A1–A3. Rack extents were *derived*
from the constraint that both aisles stay clear, because the labelled points
taken as centres of 5 m racks put Rack 2 on top of the x=0 aisle. J1 is not
in the source drawing; assumed to be (0,0). Do not treat these as ground
truth.

**Known design issue:** the central aisle measures 2.00 m; two robots at
0.98 m overall width need ~2.36 m to pass. One robot fits, two do not.

## Style

Match what's there: type hints, dataclasses, docstrings that say *why* not
*what*, numpy only. No new dependencies without asking.
