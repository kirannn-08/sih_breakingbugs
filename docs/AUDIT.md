# Full audit — logic, learning, codebase

Every finding below was **reproduced by running the code**, not inferred from
reading it. Where a claim survived scrutiny that is stated too, because "we
checked and it holds" is a result.

---

## 0. The one systemic finding

```
tests exercising the full Simulation loop : 7
tests exercising components in isolation  : 44
```

Every dead component in this repo's history had a **passing unit test**:

| component | unit-tested | wired into `Simulation` |
|---|---|---|
| consensus auction (pre-fix) | yes | **no** |
| `WaitFor` / deadlock graph | yes | **no** |
| `DeadZone` | yes | **no** |
| learned policy | yes | **no** |

The suite validates *units*, not *wiring*. That is precisely how four separate
headline mechanisms stayed disconnected while the suite went green. The
guardrail in `CLAUDE.md` — "a guard that never fires is worse than no guard" —
is correct but under-enforced: nothing checks that a component is *reachable*
from `Simulation.step()`.

**Recommendation.** For every mechanism that appears in a claim, add one
integration test asserting it *fires* during a real run — a counter, a message
on the wire, a state transition. Component correctness is necessary and has
never been the failure mode here.

---

## 1. LOGIC — audited, and it holds

The coordination mathematics is the strongest part of this codebase.

| invariant | test | result |
|---|---|---|
| L2 safety can only ever *reduce* speed | 6560 live calls | **0 violations** |
| Stopping is computed, never a reflex | 20 000 random states | **0 stops above `v_min`** |
| `resolve_auction` is order-independent | all 24 key permutations | **1 winner** |
| `heuristic()` is admissible | 40 planned paths | **0 inadmissible** |
| No `Robot` reads a peer directly | AST sweep of `sim2d.py` | **clean** |

No `Robot` method touches `.robots`. Robot-to-robot isolation is genuinely
intact; all peer knowledge arrives through `comms.py`.

### 1.1 But the task state machine runs on the server — **HIGH**

The isolation that holds robot-to-robot does **not** hold server-to-robot.
AST sweep of `Simulation`:

```
resolve_deadlocks()
    writes : blocked_since, goal, saved_goal, yielding, yield_deadline
    invokes: priority, replan
step()
    writes : task, goal, path, mode, payload, phase, saved_goal,
             yielding, blocked_since, last_broadcast
```

`Simulation.step()` writes **ten** robot attributes. The pickup→dropoff
transition, payload loading, parking and deadlock yielding are all decided
centrally and written onto the robot object. `resolve_deadlocks()` builds one
fleet-wide `DeadlockDetector`, picks the yielder and assigns its goal.

This is the same class of defect as the auction was before BUG 8, and it is
the honest answer to "where does the computation happen?": **everything except
the task state machine and deadlock resolution runs on the AMR.**

It also explains why deadlock resolution is weak. `DeadlockDetector` and the
`WaitFor` message were designed for each robot to build the graph from
broadcasts. `WaitFor` is still never sent, so the central detector is the only
implementation, and it resolves 0 of ~2800 detected cycles.

**Fix.** Move the phase machine into `Robot` (it already owns `phase`), and
broadcast `WaitFor` so each robot elects the same yielder locally.

---

## 2. LEARNING

### 2.1 The learned policy is never used — **CRITICAL**

`use_policy` flows `Simulation.__init__` → `flags["policy"]` and **is never
read**. Only `flags["congestion"]` and `flags["speed_adapt"]` are consumed.
`HybridCoordinator` — the class whose entire job is to arbitrate between the
policy and the rule — is instantiated **only in tests**.

Decisive check, per this repo's own guardrail:

```
Simulation(seed=3, use_policy=False) vs (use_policy=True)
  avg_task_time  32.17  ==  32.17
  collisions         0  ==      0
  tasks, distance, replans, deadlocks — all identical
```

Every behavioural metric is identical. The only difference in the whole
result dict is `comms.bytes`, and that is caused by an unrelated bug (§3.1),
not by the policy.

So the "Edge-AI" component is **fully disconnected**. `policy.npz` is trained,
saved, loaded by `demo_scenarios.py` and the dashboard, reported on — and
never allowed to influence a single robot decision.

### 2.2 ...but its accuracy claim is honest, and understated

I expected leakage: the split in `train_policy.py` is random over *samples*,
and with ~24 samples per trajectory, states from one path can land on both
sides. Tested against a split that holds out **whole trajectories**:

```
repo split (random over samples) : 0.918      <- the claim
honest split (whole trajectories): 0.944
majority-class baseline          : 0.296
```

The honest split scores **higher**. There is no leakage inflation, and the
lift over guessing the commonest action is **+64.8 points**. The 91.8% figure
is real and conservative. Quote it with confidence.

The gap is wiring, not science. Wiring it requires an action interface that
does not exist: robots follow a planned path, they do not consume the policy's
discrete 5-action output. That is why it was never connected, and it is a real
piece of work, not a one-line flag.

### 2.3 `rl/` scaling study

Separate folder, isolated, imports nothing from production. Two results stand:
classical WHCA\* holds within 10% of peak from 32 to 192 agents and only
collapses past ~21% density; behaviour cloning reaches 87.9% action agreement
but 4–45% of WHCA\*'s throughput. Both reported as measured, the second as a
negative result. No issues found.

---

## 3. CODEBASE

### 3.1 `amr_msgs._seq_counters` is un-resettable global state — **MEDIUM**

A module-level dict that `make_header()` mutates and nothing ever clears. It
persists across `Simulation` instances, so **the same seed does not produce
the same byte counts**:

```
same config, run twice -> comms.bytes 4 828 239 then 4 837 095
```

`test_deterministic_given_seed` does not catch this because it checks only
`tasks_completed` and `distance_m`. In the long-lived dashboard process the
counters grow without bound for the life of the server.

**Fix.** Make the counter an instance of the mediator, or expose a reset and
call it from `Simulation.__init__`. Then widen the determinism test.

### 3.2 The frozen contract has drifted — **MEDIUM**

Declared in `amr_msgs.py` and referenced nowhere in the running system:

| symbol | note |
|---|---|
| `Telemetry` | documented as *"Dashboard team: build against this and nothing else"* — the dashboard builds its own ad-hoc dict instead |
| `WaitFor` | the missing input to distributed deadlock resolution |
| `Coordination` | speed negotiation message |
| `Obstacle`, `Reroute` | never sent |
| `is_higher_priority` | dead helper |
| `Reservation.overlaps` | dead helper — the planner reimplements the same interval test inline |
| `Header.is_stale` / `age` | exercised only by tests, so **TTL is never enforced at runtime** |
| `resolve_by_priority` (`coordination.py`) | dead — the rule is inlined twice in `sim2d.py:454,481` |

A contract nothing is checked against is documentation, not a contract. Either
wire these or mark them explicitly as reserved-for-ROS2 so the next reader
does not assume they are live.

### 3.3 Smaller items

- `Simulation` still caps at 5 robots (`starts` list, `IndexError` at 6).
- `run_sih_layout.py` still raises `TypeError` before running; the +20.0% SIH
  figure remains unreproducible.
- Open HIGH items from `SECURITY_AUDIT.md` are unchanged: malformed-packet
  crash, unbounded bid values, unauthenticated `Claim`/`Release`, uncapped
  `Intent`.

---

## 4. What I would do next, in order

1. **Wire or retire the learned policy.** It is the single largest gap between
   what this project claims and what it runs. If wiring is too big before the
   deadline, say plainly that the policy is trained and benchmarked but not
   yet in the control loop — that is a defensible position; an unexamined
   "Edge-AI" claim is not.
2. **Move the task state machine and deadlock resolution onto the robot.**
   Closes the last gap in the central claim and is prerequisite to fixing the
   terminal freeze.
3. **Add reachability tests** for every claimed mechanism (§0).
4. **Reset `_seq_counters`** and widen the determinism test.
5. Flaw 2 (space-time A\*) — and note B1 shows congestion routing is currently
   worth nothing, which is consistent with that flaw being open.
