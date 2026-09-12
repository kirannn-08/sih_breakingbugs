# Deadlock Resolution, Predictive Avoidance, and Sensor-Driven Rerouting

Design note and results for the work that replaced priority-based yielding
with a LIFO contested-region stack, gave the LiDAR a decision role, put time
into the planner's state, and trained a deadlock-risk model on this
warehouse.

Everything below was measured on `WarehouseMap`, 4 robots, 180 s, 20 seeds,
with occluded sensing. Reproduce with:

```bash
python3 run_benchmark.py 20      # results/benchmark.csv, one row per arm per seed
python3 synth_data.py            # regenerates data/ (~5 min, gitignored)
python3 train_deadlock.py        # results/deadlock_model.csv
cd rl && python3 train.py && python3 run_study.py   # results/mapf_scaling.csv
```

---

## 1. The problem

Four AMRs, 0.98 m wide, in aisles 1.50 m wide. Two robots cannot pass in an
aisle; the central aisle of the SIH layout cannot fit two at all. So a head-on
meeting inside a corridor has **no velocity solution** — no reciprocal
avoidance method, ORCA included, can solve it, because the feasible velocity
set is empty. The only options are to prevent the meeting or to reverse out
of it.

Measured starting state, seed 3: all four robots wedged in the x=28..30 aisle,
a wait-for 2-cycle persisting **4589 robot-ticks**, robots stalled ~75% of the
run, **nine detected deadlocks and two completed tasks**.

Detection was never the problem. Resolution was.

---

## 2. Why priority-based yielding could not work

`DeadlockDetector.choose_yielder` elects the lowest-priority member of the
cycle. Priority is built from task class, slack, battery and id — none of
which correlates with *who is able to move*.

So it kept electing a robot buried at the far end of an aisle whose only exit
ran through the peer it was deadlocked with. The yielder replanned to a
retreat cell, A\* routed it back through the jam, and the jam survived.

Two further defects compounded it:

| defect | measured consequence |
|---|---|
| election ran over the **cycle only** | robot 2 unanimously elected 1147×, failed to execute **1146×** — robot 1, merely *queued* behind the cycle and never a member of it, was parked in its reverse path |
| escape was a **replan** | A\* routes to the retreat cell by cheapest cost, which in a jam is straight back through the deadlock |

Robot 1 had entered the aisle at t=68.2 against robot 2's t=48.8. It was the
true top of the stack. Electing from the cycle alone could not see it.

---

## 3. Approach: the contested region is a stack

**Entering a contested region pushes. Resolving pops. It pops from the top.**

The last robot into a corridor is, by construction, the one nearest the
entrance. Its escape route is the stretch it has just driven, and nobody is
standing in it — anyone behind it entered later still and has already been
popped. That property is what makes the manoeuvre executable, and priority
ordering has no equivalent.

This is **timestamp deadlock resolution**: the *wait-die* rule of Rosenkrantz,
Stearns & Lewis (1978). The younger transaction aborts and retries, the older
one proceeds. It is starvation-free for the same reason theirs is — an entry
stamp only ages relative to new arrivals, so no robot can be elected to yield
forever.

### 3.1 Election order

`DeadlockDetector.choose_yielder_lifo(stack, entered_at, dist_to_block, priorities)`

1. **latest `entered_at`** — the top of the stack.
2. then **shortest distance to the blocking robot**. Two robots entering from
   opposite ends within `LIFO_EPS` have no meaningful stack order, so the tie
   goes to whoever is closest to the pinch point: least corridor to clear,
   soonest release. *(This is the "one nearer the deadlock reroutes" rule.)*
3. then lowest priority, then highest id — total and deterministic, so the
   answer is never ambiguous.

A robot whose `WaitFor` was lost reports `entered_at = 0.0` and therefore
sorts as the **oldest** entrant. Missing evidence must never volunteer a
victim: a dropped packet cannot elect someone to back out.

Every robot computes this from broadcast data and reaches the same answer, so
there is no negotiation round-trip and no referee. Verified live: both cycle
members elected the same robot 1147 and 1148 times out of 1148.

### 3.2 The stack is the cycle *plus its waiting tree*

`DeadlockDetector.blocked_set` closes the wait-for graph inward: the cycle
plus the transitive closure of edges leading into it. A cycle names the
robots that are deadlocked; it does not name the robots whose bodies are in
the way of resolving it. Popping the latest entrant of the *whole* set frees
the next one down, and the stack unwinds in reverse entry order.

### 3.3 Backout, not replan

`backout_target` reverses along the robot's own **trail** — the cells it has
actually occupied — to the first cell wide enough for a peer to pass. The
trail is the one route known to be physically drivable and, for the last
robot in, the one route no cycle member occupies. Backing out along a
previously traversed path is the same device used by Čáp, Gregoire & Frazzoli
(2016) for provably deadlock-free execution under delays.

### 3.4 Prediction: don't enter a jam you can foresee

`corridor_blocked` only ever asked who is inside the aisle **right now**,
which is too late — two robots converging on opposite mouths both see an
empty corridor, both enter, and no velocity solution exists once they are in.
Every wedge measured on this map formed exactly that way.

`segment_entry_deferred` derives from the `Intent` path cells peers **already
broadcast** when each will be inside each segment and which way
(`segment_windows` → `predicted_head_on`). An opposed overlap is therefore
visible while both robots are still outside and either can still stop
cheaply. The later entrant defers. Same stack discipline, applied before the
push instead of after.

No new message type — this is the `Intent` that was already on the air. It is
deadlock *avoidance* in Dijkstra's sense (refuse a state known to lead to an
unsafe one) layered above deadlock *recovery*.

The two entry rules are evaluated **without short-circuiting**: an `or` would
leave the predictive rule uncounted whenever the reactive one fired first,
and an uncounted rule is indistinguishable from a dead one.

---

## 4. LiDAR as a decision input, not just a brake

Previously LiDAR fed only the geometric brake and `orca_adjust`. Every
*decision* came from radio. Two consequences: a robot with a dead radio made
no coordination decisions at all, and a stopped robot that never *sent* an
Obstacle message (crashed, flat battery, dropped pallet, a person) was
invisible to routing while in plain view of the scanner.

Sensing was also **omniscient** — returns were pure Euclidean distance, so
robots detected peers through solid racking. `WarehouseMap.line_of_sight`
now walks the grid exactly (Amanatides & Woo, 1987) and occluded returns are
dropped.

| layer | what it does | comms needed |
|---|---|---|
| `perception.LidarTracker` | id-less tracks, EMA velocity, nearest-neighbour association | none |
| `Robot.lidar_scan` | a stationary in-lane track on my own path → marked blocked, rate-limited replan | none |
| `Robot.lidar_speed_cap` | a closing in-lane track → speed cap by whichever mechanism the arm under test uses | none |
| blind-corner rule | aisle mouths (116 cells, 14% of free space) → sight-limited speed | **fused** |

Tracks carry no `robot_id`. A real LiDAR returns points, and attaching ids
would smuggle comms knowledge back into the sensor path and recreate the
omniscience being removed.

**Turning occlusion on alone produced 182 collisions on seed 1**, because two
robots meeting at an aisle mouth are invisible to each other until roughly one
body length apart — already too late for a 1.00 m robot.

The fix is the fusion: geometry says *where* a surprise is possible, the radio
says whether any peer is positioned to supply one, and the LiDAR overrides the
radio when it can already see that peer. Ungated the rule fired on **58% of
all driving ticks** and throughput collapsed to 1–5 tasks per run; gated it
fires ~100 ticks per run. With no fresh peer reports it reverts to slowing at
every mouth: **losing the radio costs speed, never safety.**

One further finding: occlusion is a *long-range optical* property. Two robots
sat diagonally across a rack corner, mutually invisible, grazing at 0.974 m
against a 0.98 m footprint — 1359 contact ticks on seed 5. A rack corner
cannot hide a body whose centre is 0.97 m from yours; the footprints already
overlap. Peers inside `D_NEAR` are now always sensed.

---

## 5. Time in the planner's state

`planner.plan`'s closed set was `set[tuple[int, int]]` — purely spatial —
while the cost function was time-dependent, since `t_arrive` drives the
occupancy lookup. The first expansion of a cell therefore won permanently, at
whatever time it happened to be reached.

The search could not represent *"wait here 3 s, then go straight through"*.
Its only way to express avoidance was to go **around**. That is the rerouting
churn: the planner had exactly one tool and used it for everything. It also
explains the standing result that congestion-aware routing alone is worth
nothing — a time-dependent cost driven by a timeless search buys detours and
nothing else.

State is now `(x, y, t_bucket)` at 1 s resolution with an explicit WAIT
successor — space-time A\* in the sense of Silver (2005). `heuristic` is
unchanged, still `distance / v_max` with no congestion term: with WAIT
available an inadmissible *h* returns arbitrarily bad plans, not merely
suboptimal ones.

Three details that decide whether this is real or decorative:

- `plan` still returns strictly adjacent cells (contract unchanged);
  `plan_st` returns the space-time path and `schedule_of` the departures.
- `Robot.replan` keeps the **schedule**. Collapsed to cells, "hold 3 s then
  go" and "go now" are identical — executing the former as the latter is how
  the planner's waits stayed invisible.
- A planned wait executes as a computed **slowdown** via `adapt_speed`,
  reaching zero only below `v_min`. Invariant 4 holds: stopping stays a
  computed last resort.

Holds are recorded only where the plan genuinely waits. Enforcing the whole
schedule would pin every robot to `v_nom` for its entire route and silently
delete speed adaptation.

Confirmed live: 1264 `plan_st` calls produced 215 WAIT actions across 147
distinct plans. Replan latency 16.2 ms for the longest route, against a
500 ms budget.

---

## 6. The learned model

### 6.1 What it predicts, and where it sits

`P(this robot is wedged within 10 s)` from its own local observation. **Not an
action.** `ARCHITECTURE.md` §0.2 already concluded that full MARL policy
control does not belong on the critical path and that the defensible learned
component is "a small supervised learned congestion/ETA predictor running
onboard, plus an optional EPH-style deadlock-escape policy behind a
rule-based fallback". This is the first of those.

It is **L4 advisory and nothing below**. It can make a robot wait at the mouth
of a corridor. It cannot command velocity, raise a speed, override the safety
supervisor, or take any part in deadlock *resolution* — that stays
deterministic LIFO, because a learned component that resolves deadlocks
*sometimes* is worse than a rule that resolves them *always*. Its failure mode
is a robot that waits unnecessarily: throughput, never safety.

### 6.2 Synthetic data, from this warehouse

`synth_data.py` rolls out `sim2d.Simulation` on `WarehouseMap` — same map,
same footprint, same safety supervisor, same occluded LiDAR the deployed model
will see. Domain randomisation is over what genuinely varies at run time
(fleet size 2–6, packet loss 0–20%, task interval 4–12 s, seed), **not over
the building**. 60 runs → **58,763 samples, 57.3% positive, 62 features**.

Labels are **outcomes, not a heuristic**: positive if that robot actually
became wedged within 10 s — entered a wait-for cycle, or stalled past
`T_STALL` *while still having somewhere to go*. Labelling with a rule would
teach the model to imitate the rule, and the rule is already free. The "still
having somewhere to go" clause matters: without it, a third of the positives
are robots parked on charging bays and the model learns to predict parking.

`features.py` is called by **both** the generator and `sim2d` at run time.
Writing the features twice is the standard way to ship a model that scores 90%
offline and does nothing online.

### 6.3 Results — and why the headline number is the small one

Split **by run**, not by sample: samples are 0.5 s apart with 10 s label
horizons, so a random split puts near-duplicates of test rows into training.

| variant | acc | prec | rec | F1 | AUC |
|---|---|---|---|---|---|
| **model, early warning (moving robots)** | **0.909** | **0.780** | **0.475** | **0.591** | **0.839** |
| model, already stalled | 0.991 | 0.991 | 1.000 | 0.995 | 0.706 |
| model, all held-out | 0.948 | 0.973 | 0.931 | 0.951 | 0.975 |
| model, train (all) | 0.949 | 0.978 | 0.934 | 0.955 | 0.982 |
| baseline: majority class | 0.546 | 0.546 | 1.000 | 0.706 | 0.500 |
| baseline: hand-written rule | 0.455 | 1.000 | 0.001 | 0.002 | 0.501 |
| **leakage check: `stall_age` alone** | — | — | — | — | **0.930** |

6,178 params, 0.007 ms inference, trains in 0.8 s. Train/test gap 0.001 — no
memorisation.

**Do not quote 94.8% / AUC 0.975.** Half the evaluation set is robots that are
already stalled, that slice is 99.1% positive, and `stall_age` on its own
scores AUC 0.930 — most of the apparent skill is the model restating a stall
that has already happened. The operating regime is a *moving* robot deciding
whether to enter a corridor, and there the honest figure is **AUC 0.839,
recall 0.475 at precision 0.780 on a 13.8% base rate**.

Training on the moving subset alone was tried and scores identically
(AUC 0.839), so the shipped model trains on everything.

The hand-written baseline is worth a note: *"narrow aisle or blind corner AND
a peer in my segment heading at me"* has recall **0.001**. Its opposed-heading
test needs a peer to be *moving* toward you, and in a forming jam everyone has
already slowed. The obvious rule cannot see stopped robots. That is the gap
the model is filling.

### 6.4 In the loop, the model does not pay for itself

Threshold swept in closed loop, 10 seeds × 180 s:

| threshold | tasks completed | avg task time |
|---|---|---|
| rule only | 60 | 33.36 s |
| 0.80 | 54 | 32.26 s |
| 0.90 | 57 | 33.29 s |
| 0.95 | 59 | 33.89 s |
| 0.99 | 62 | 34.56 s |

At 0.99 it is a wash (+2 tasks of 60, inside seed noise; paired *t* = 1.45
over 20 seeds); everywhere below, a clear loss. **The model has genuine
offline skill and does not convert it into throughput.** It is therefore
`HIGH_CONF = 0.99` and **off by default** (`use_risk_model=False`), and this
is reported rather than tuned away.

### 6.5 The MARL policy does not work, now demonstrated on our map

`rl/` previously trained on a generic 40×40 rack grid sharing no geometry with
this warehouse — different aisle widths, no passing bays, no blind corners, a
different task distribution. `sih_warehouse_grid()` now builds the MAPF grid
from `WarehouseMap` itself, and `train.py`/`run_study.py` use it at matched
density.

Distilled from WHCA\* (Silver 2005), **87.1% held-out action agreement**:

| N | density | greedy+jitter | WHCA\* | learned | learned / WHCA\* |
|---|---|---|---|---|---|
| 8 | 0.9% | 0.427 | 2.948 | 0.958 | **33%** |
| 16 | 1.7% | 0.281 | 3.208 | 0.375 | 12% |
| 32 | 3.4% | 0.302 | 2.875 | 0.310 | 11% |
| 64 | 6.8% | 0.263 | 3.003 | 0.167 | 6% |
| 128 | 13.6% | 0.224 | 2.572 | 0.073 | **3%** |

87% per-step agreement collapses to 3–33% of the expert's throughput, and the
policy is beaten by greedy-with-jitter at every N ≥ 16. This is textbook
compounding covariate shift in behavioural cloning: per-step accuracy does not
compose over a 400-step episode, and the states the policy reaches are not the
states it was trained on.

**So: the answer to "is the RL/EPH/multi-agent side working?" is no.** It runs,
on the right map, with honest numbers — and it loses to a classical planner.
That is a result worth having rather than a gap worth hiding. EPH-style
ensembling exists to attack exactly this gap and is not implemented here; the
citation in `README.md` describes prior art, not this code.

---

## 7. Benchmark

4 robots, 180 s, 20 seeds, occluded sensing. `results/benchmark.csv` carries
one row per arm per seed with every counter.

| arm | thru/min | tasks | task_s | coll | vs B0 (task time) |
|---|---|---|---|---|---|
| B0 stop-and-wait | 1.55 | 4.7 | 43.4 | 0 | baseline |
| B1 + congestion routing | 1.57 | 4.7 | 43.6 | 0 | −0.7% |
| B2 + speed adaptation | 1.75 | 5.2 | 33.9 | 0 | **+21.8%** |
| B3 full | 2.02 | 6.0 | 36.5 | 0 | +15.9% |
| B4 full @ 20% loss | 1.73 | 5.2 | 34.3 | 0 | +20.8% |
| B5 full, LIFO stack off | 1.82 | 5.5 | 36.0 | 0 | +17.0% |
| B6 full + risk model | 2.05 | 6.2 | 37.0 | 0 | +14.5% |

Paired per-seed comparisons, n = 20:

| comparison | mean Δ tasks | t | verdict |
|---|---|---|---|
| full vs stop-and-wait | +1.40 | 2.55 | **significant** |
| LIFO stack on vs off | +0.60 | 1.75 | not established |
| risk model vs rule only | +0.10 | 1.45 | not established |
| congestion alone vs B0 | +0.05 | 0.12 | indistinguishable |
| speed adaptation alone vs B0 | +0.60 | 1.20 | not established |

### 7.1 The two metrics disagree, and the disagreement is real

`avg_task_time` averages over **completed** tasks only, so a system that
finishes the hard tasks its baseline abandons is *penalised for finishing
them* — the extra completions are the slow ones. Seed 3: baseline 1 task,
full system 5.

Throughput has no such bias: **4.7 → 6.0 tasks per run, +30.1%, t = 2.55.**

`test_T18_beats_stop_and_wait` asserts ≥20% on `avg_task_time` and **fails at
+15.9%**. It is left failing. Per `CLAUDE.md`, the fix or the expectation is
wrong, not the test — and the case that the expectation is measuring the wrong
quantity is made above, not by editing the assertion.

B1 remains worth nothing (−0.7%) **even after** time entered the planner's
state, so the earlier diagnosis was incomplete: congestion-aware routing on
this map does not pay for itself regardless.

### 7.2 Every new rule fires

Totals, B3 across 20 seeds. A rule whose counter is always zero is dead code
in a costume — two such bugs are already in `ENGINEERING_LOG.md`.

| counter | total | meaning |
|---|---|---|
| `backouts` / `backouts_done` | 84 / 82 | LIFO pops started / completed |
| `entry_deferrals` | 2523 | predicted head-on refused entry |
| `blind_slowdowns` | 2127 | sight-limited speed at an aisle mouth |
| `lidar_blocks` | 1559 | cells marked blocked from sensing alone |
| `lidar_replans` | 96 | reroutes triggered with no message received |
| `planned_waits` | 97 | WAIT actions the planner asked for |
| `collisions` | **0** | across all 140 runs, all arms |

Deadlock recovery went from *never* to 1.4–20 s.

---

## 8. What is still wrong

- **`test_T18` fails at +15.9%** against a 20% target on a metric argued above
  to be the wrong one. Not silently fixed.
- **The LIFO stack's throughput gain is not statistically established**
  (t = 1.75, n = 20). It demonstrably resolves jams that previously never
  resolved; that it raises throughput is suggested, not proven. More seeds
  would settle it.
- **The risk model does not earn its place in the loop** and ships disabled.
- **The MARL policy loses to a classical planner** at every density tested.
- **`use_policy` is still dead** (`DEAD_CODE.md` Tier 3). This work added
  `use_risk_model`, which is consumed; the action policy remains unreachable.
- **Flaw 3 (instant heading snap) and flaw 5 (`orca_adjust` is not ORCA)** from
  `CLAUDE.md` are untouched. Rotation is still free, so broadcast ETAs remain
  optimistic — and ETA error is what desynchronises conflict prediction on
  real hardware.
- **`run_sih_layout.py` is still broken**, so the SIH-layout figure remains
  unreproducible.
- Sensor noise is not modelled. `test_deterministic_given_seed` is what makes
  the benchmark reproducible; noise belongs in the ROS 2 / Gazebo transfer,
  where a real driver supplies it.

---

## 9. References

Deadlock theory
- Coffman, Elphick & Shoshani (1971). *System Deadlocks.* ACM Computing
  Surveys 3(2). — the four necessary conditions; circular wait is the one
  attacked here.
- Rosenkrantz, Stearns & Lewis (1978). *System Level Concurrency Control for
  Distributed Database Systems.* ACM TODS 3(2). — **wait-die**, the timestamp
  rule §3 implements.
- Dijkstra (1965). *Cooperating Sequential Processes.* — deadlock avoidance by
  refusing unsafe states; the framing for §3.4.

Multi-agent path finding
- Silver (2005). *Cooperative Pathfinding.* AIIDE. — WHCA\*, space-time A\*
  with a reservation table (§5, and the expert in §6.5).
- Standley (2010). *Finding Optimal Solutions to Cooperative Pathfinding
  Problems.* AAAI.
- Sharon, Stern, Felner & Sturtevant (2015). *Conflict-Based Search.* AIJ.
- Ma, Li, Kumar & Koenig (2017). *Lifelong Multi-Agent Path Finding for Online
  Pickup and Delivery Tasks.* AAMAS. — the lifelong formulation in `rl/env.py`.
- Čáp, Gregoire & Frazzoli (2016). *Provably Safe and Deadlock-Free Execution
  of Multi-Robot Plans under Delaying Disturbances.* IROS. — backing out along
  a traversed path (§3.3).
- Stern et al. (2019). *Multi-Agent Pathfinding: Definitions, Variants, and
  Benchmarks.* SoCS.

Learning
- Sartoretti et al. (2019). *PRIMAL.* RA-L. — imitation from a centralised
  planner into a decentralised local policy.
- Tang, Berto & Park (2024). *EPH: Ensembling Prioritized Hybrid Policies for
  Multi-agent Pathfinding.* — **prior art, not implemented here.** Cited
  because it targets precisely the closed-loop collapse measured in §6.5.
- Ross, Gordon & Bagnell (2011). *A Reduction of Imitation Learning and
  Structured Prediction to No-Regret Online Learning (DAgger).* AISTATS. —
  the compounding-error analysis §6.5 is an instance of.

Geometry and control
- Amanatides & Woo (1987). *A Fast Voxel Traversal Algorithm for Ray Tracing.*
  Eurographics. — `line_of_sight` (§4).
- van den Berg, Guy, Lin & Manocha (2011). *Reciprocal n-Body Collision
  Avoidance.* ISRR. — **ORCA proper.** `Robot.orca_adjust` is *not* this; it
  is a potential field with a fixed sidestep convention. `CLAUDE.md` flaw 5,
  still open.
