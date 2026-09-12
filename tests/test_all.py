"""
Test suite  --  maps to the T1..T18 plan in the architecture doc (section K).

Run:  python3 -m pytest tests/ -v
"""

from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from amr_msgs import (INF_COST, Header, LinkPacket, MsgType, RobotState,
                      make_header, priority_tuple, wrap)
from comms import CommsMediator, DeadZone
from coordination import (DeadlockDetector, adapt_speed, bid_cost, feasible,
                          resolve_auction)
from planner import ReservationTable, plan, path_to_reservations
from sim2d import Simulation
from warehouse_map import WarehouseMap


# ==========================================================================
# MAP
# ==========================================================================

class TestMap:
    def test_map_builds(self):
        m = WarehouseMap()
        assert m.w == 40 and m.h == 30

    def test_has_passing_bays(self):
        """Deadlock escape is IMPOSSIBLE without somewhere to yield to."""
        m = WarehouseMap()
        assert len(m.passing_bays) >= 10

    def test_map_is_adversarial(self):
        """A map where robots rarely meet produces a meaningless number."""
        m = WarehouseMap()
        s = m.stats()
        assert s["narrow_fraction"] > 0.10, "not enough narrow aisles to force conflict"

    def test_all_nodes_reachable(self):
        m = WarehouseMap()
        table = ReservationTable()
        nodes = list(m.nodes.values())
        for a in nodes[:4]:
            for b in nodes[4:7]:
                p = plan(m, (a.cx, a.cy), (b.cx, b.cy), table, 1, 0, 0)
                assert p, f"{a.name} -> {b.name} unreachable"

    def test_aisle_segments_span_width(self):
        """REGRESSION (bug #3): segmentation must flood-fill ACROSS the
        corridor width. Per-column segmentation silently disabled the
        corridor reservation rule."""
        m = WarehouseMap()
        # Moved from row 21 to row 19. Row 21 carries the P2 destination
        # pocket, so that stretch is now 5 cells / 2.50 m wide and is
        # correctly NOT a narrow corridor any more -- aisle_at is -1 there by
        # design. Row 19 is the same aisle at full 3-cell width, so the
        # property under test (flood fill spans the corridor WIDTH, not one
        # column) is unchanged. The assertion is not relaxed.
        a = m.aisle_at(14, 19)
        assert a != -1
        assert m.aisle_at(15, 19) == a, "same corridor split across columns"
        assert m.aisle_at(16, 19) == a, "same corridor split across columns"
        # and the widened rows really are open, which is the point of widening
        assert m.aisle_at(15, 21) == -1, "P2 pocket should not be a narrow aisle"


# ==========================================================================
# MESSAGES
# ==========================================================================

class TestMessages:
    def test_roundtrip(self):
        st = RobotState(header=make_header(1), x=1.0, y=2.0, theta=0.5)
        pkt = wrap(st, MsgType.ROBOT_STATE, 1)
        back = LinkPacket.from_json(pkt.to_json())
        assert back.src == 1
        assert back.payload["x"] == 1.0

    def test_staleness(self):
        h = Header(robot_id=1, seq=1, stamp=100.0, ttl_ms=2000)
        assert not h.is_stale(101.0)
        assert h.is_stale(103.0)

    def test_priority_is_total_and_deterministic(self):
        """Every robot must derive the SAME ordering from broadcast data,
        or decentralised prioritised planning diverges."""
        a = priority_tuple(1, 10.0, 0.9, 1)
        b = priority_tuple(1, 10.0, 0.9, 2)
        assert tuple(a) < tuple(b)          # id breaks the tie
        assert priority_tuple(1, 5.0, 0.9, 3) > priority_tuple(1, 20.0, 0.9, 3)


# ==========================================================================
# T1-T4  TASK ALLOCATION
# ==========================================================================

class TestAuction:
    def test_T1_lowest_cost_wins(self):
        bids = {1: (25.0, True), 2: (11.0, True), 3: (18.0, True)}
        assert resolve_auction(bids) == 2

    def test_T1b_deterministic_tiebreak(self):
        """No auctioneer exists, so ties must resolve identically on every
        robot without extra messages."""
        bids = {3: (10.0, True), 1: (10.0, True), 2: (10.0, True)}
        assert resolve_auction(bids) == 1
        for _ in range(50):
            assert resolve_auction(bids) == 1

    def test_T2_capacity_gate(self):
        assert not feasible(capacity_kg=10, payload_kg=0, task_kg=18,
                            battery_soc=1.0, energy_needed=5, energy_full=100)
        assert feasible(capacity_kg=20, payload_kg=0, task_kg=18,
                        battery_soc=1.0, energy_needed=5, energy_full=100)

    def test_T2b_infeasible_never_wins(self):
        bids = {1: (5.0, False), 2: (99.0, True)}
        assert resolve_auction(bids) == 2

    def test_T3_battery_beats_proximity(self):
        """The headline allocation case: a near-but-depleted robot must LOSE
        to a far-but-charged one. Falls out of the energy-FRACTION term with
        no special-case rule."""
        near_empty = bid_cost(eta=8, congestion=0, energy_needed=20,
                              energy_available=8, queue_len=0, conflict_risk=0)
        far_full = bid_cost(eta=25, congestion=0, energy_needed=20,
                            energy_available=80, queue_len=0, conflict_risk=0)
        assert far_full < near_empty

    def test_T4_no_feasible_bidder(self):
        assert resolve_auction({1: (INF_COST, False)}) == 0
        assert resolve_auction({}) == 0


# ==========================================================================
# PLANNER
# ==========================================================================

class TestPlanner:
    def test_path_valid_and_connected(self):
        m, t = WarehouseMap(), ReservationTable()
        p = plan(m, (2, 2), (37, 27), t, 1, 0, 0)
        assert p and p[0] == (2, 2) and p[-1] == (37, 27)
        for a, b in zip(p, p[1:]):
            assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
            assert m.is_free(*b)

    def test_unreachable_returns_empty(self):
        m, t = WarehouseMap(), ReservationTable()
        assert plan(m, (2, 2), (4, 8), t, 1, 0, 0) == []   # inside a shelf

    def test_congestion_changes_route(self):
        """If peer reservations don't alter the chosen path, the congestion
        term is not actually wired in."""
        m = WarehouseMap()
        free_t = ReservationTable()
        base = plan(m, (2, 14), (37, 14), free_t, 1, 0, 0)

        busy = ReservationTable()
        res = [r for r in path_to_reservations(base, 0.0, 0.6, horizon=999)]
        for pid in range(2, 6):
            busy.update(pid, res, 0.0)
        alt = plan(m, (2, 14), (37, 14), busy, 1, 0, 0)
        assert alt, "congestion made the goal unreachable"
        assert alt != base or len(res) < 5

    def test_blocked_belief_decays(self):
        """Temporary obstructions must self-clear, or the map degrades
        monotonically over a long run."""
        t = ReservationTable()
        t.mark_blocked(5, 5, now=0.0, confidence=1.0)
        assert t.blocked_belief(5, 5, 0.0) == pytest.approx(1.0)
        assert t.blocked_belief(5, 5, 10.0) == pytest.approx(0.5)
        assert t.blocked_belief(5, 5, 100.0) == 0.0

    def test_stale_peer_reservations_decay(self):
        t = ReservationTable()
        t.update(2, path_to_reservations([(5, 5), (5, 6)], 0.0, 0.6), 0.0)
        fresh = t.occupancy(5, 5, 0.0, now=0.0)
        stale = t.occupancy(5, 5, 0.0, now=30.0)
        assert fresh > 0 and stale < fresh

    def test_replan_latency_budget(self):
        """Architecture doc claims 1-8 ms on a Pi 5. Assert generously."""
        import time
        m, t = WarehouseMap(), ReservationTable()
        t0 = time.time()
        for _ in range(20):
            plan(m, (2, 2), (37, 27), t, 1, 0, 0)
        ms = (time.time() - t0) / 20 * 1000
        assert ms < 500, f"replan {ms:.1f} ms is too slow even for a laptop"


# ==========================================================================
# T5-T7  SPEED ADAPTATION  (the headline mechanism)
# ==========================================================================

class TestSpeedAdaptation:
    def test_T5_slows_instead_of_stopping(self):
        """The whole thesis: conflict -> SLOW, not STOP."""
        v, action = adapt_speed(dist_to_conflict=4.0, peer_exit_time=8.0,
                                now=0.0, v_cur=0.8, v_min=0.15, v_max=0.8)
        assert action == "SLOW"
        assert 0.15 <= v < 0.8

    def test_T5b_stop_is_last_resort_only(self):
        """STOP must be a COMPUTED outcome, reached only when the required
        speed falls below v_min -- never a reflex."""
        v, action = adapt_speed(dist_to_conflict=0.5, peer_exit_time=60.0,
                                now=0.0, v_cur=0.8, v_min=0.15, v_max=0.8)
        assert action == "STOP" and v == 0.0

    def test_proceeds_when_clear(self):
        v, action = adapt_speed(dist_to_conflict=20.0, peer_exit_time=1.0,
                                now=0.0, v_cur=0.3, v_min=0.15, v_max=0.8)
        assert action == "PROCEED" and v == 0.8

    def test_hysteresis_prevents_chatter(self):
        """Without a deadband the commanded speed oscillates every tick."""
        v, _ = adapt_speed(4.0, 8.0, 0.0, v_cur=0.505, v_min=0.15, v_max=0.8)
        assert v == 0.505


# ==========================================================================
# T10  DEADLOCK
# ==========================================================================

class TestDeadlock:
    def test_T10_detects_3cycle(self):
        d = DeadlockDetector()
        d.set_edge(1, 2); d.set_edge(2, 3); d.set_edge(3, 1)
        assert sorted(d.find_cycle()) == [1, 2, 3]

    def test_detects_2cycle(self):
        d = DeadlockDetector()
        d.set_edge(1, 2); d.set_edge(2, 1)
        assert sorted(d.find_cycle()) == [1, 2]

    def test_no_false_positive_on_chain(self):
        """A chain is NOT a deadlock. False positives cause pointless yields."""
        d = DeadlockDetector()
        d.set_edge(1, 2); d.set_edge(2, 3)
        assert d.find_cycle() == []

    def test_yielder_is_lowest_priority_and_agreed(self):
        prios = {1: [1, 0, 0, 1], 2: [1, 0, 0, 2], 3: [9, 0, 0, 3]}
        assert DeadlockDetector.choose_yielder([1, 2, 3], prios) == 3

    def test_stall_needs_sustained_time(self):
        d = DeadlockDetector()
        assert not d.update_stall(1, speed=0.0, now=0.0)
        assert not d.update_stall(1, speed=0.0, now=2.0)
        assert d.update_stall(1, speed=0.0, now=5.0)

    def test_stall_clears_on_movement(self):
        d = DeadlockDetector()
        d.update_stall(1, 0.0, 0.0)
        d.update_stall(1, 0.6, 1.0)
        assert not d.update_stall(1, 0.0, 2.0)


# ==========================================================================
# T11-T14  COMMS  (the differentiator)
# ==========================================================================

class TestComms:
    def test_basic_delivery(self):
        c = CommsMediator([1, 2, 3])
        st = RobotState(header=make_header(1), x=0, y=0, theta=0)
        c.send(wrap(st, MsgType.ROBOT_STATE, 1), 0.0)
        c.step(1.0)
        assert len(c.receive(2)) == 1 and len(c.receive(3)) == 1

    def test_sender_never_receives_own_message(self):
        c = CommsMediator([1, 2])
        st = RobotState(header=make_header(1), x=0, y=0, theta=0)
        c.send(wrap(st, MsgType.ROBOT_STATE, 1), 0.0)
        c.step(1.0)
        assert len(c.receive(1)) == 0

    def test_T12_isolate_one_robot(self):
        c = CommsMediator([1, 2, 3])
        c.isolate(2)
        st = RobotState(header=make_header(1), x=0, y=0, theta=0)
        c.send(wrap(st, MsgType.ROBOT_STATE, 1), 0.0)
        c.step(1.0)
        assert len(c.receive(2)) == 0, "isolated robot still receiving"
        assert len(c.receive(3)) == 1, "unrelated link wrongly cut"

    def test_T13_total_blackout(self):
        c = CommsMediator([1, 2, 3])
        c.cut_everything()
        st = RobotState(header=make_header(1), x=0, y=0, theta=0)
        c.send(wrap(st, MsgType.ROBOT_STATE, 1), 0.0)
        c.step(1.0)
        assert len(c.receive(2)) == 0 and len(c.receive(3)) == 0

    def test_link_restore(self):
        c = CommsMediator([1, 2])
        c.cut_everything(); c.restore_all()
        st = RobotState(header=make_header(1), x=0, y=0, theta=0)
        c.send(wrap(st, MsgType.ROBOT_STATE, 1), 0.0)
        c.step(1.0)
        assert len(c.receive(2)) == 1

    def test_T14_packet_loss_is_partial(self):
        import random
        c = CommsMediator([1, 2], random.Random(0), loss_rate=0.5)
        for i in range(200):
            st = RobotState(header=make_header(1), x=0, y=0, theta=0)
            c.send(wrap(st, MsgType.ROBOT_STATE, 1), 0.0)
        c.step(1.0)
        n = len(c.receive(2))
        assert 60 < n < 140, f"loss model not behaving: {n}/200"

    def test_range_limit(self):
        c = CommsMediator([1, 2], comm_range=5.0)
        c.update_position(1, 0.0, 0.0)
        c.update_position(2, 50.0, 0.0)
        st = RobotState(header=make_header(1), x=0, y=0, theta=0)
        c.send(wrap(st, MsgType.ROBOT_STATE, 1), 0.0)
        c.step(1.0)
        assert len(c.receive(2)) == 0

    def test_dead_zone(self):
        import random
        c = CommsMediator([1, 2], random.Random(1))
        c.dead_zones.append(DeadZone(0, 0, 10, 10, deliver_prob=0.0))
        c.update_position(1, 5.0, 5.0)
        c.update_position(2, 20.0, 20.0)
        for _ in range(20):
            st = RobotState(header=make_header(1), x=0, y=0, theta=0)
            c.send(wrap(st, MsgType.ROBOT_STATE, 1), 0.0)
        c.step(1.0)
        assert len(c.receive(2)) == 0

    def test_latency_is_not_instant(self):
        c = CommsMediator([1, 2], latency_mean=0.5, latency_std=0.0)
        st = RobotState(header=make_header(1), x=0, y=0, theta=0)
        c.send(wrap(st, MsgType.ROBOT_STATE, 1), 0.0)
        c.step(0.1)
        assert len(c.receive(2)) == 0
        c.step(0.6)
        assert len(c.receive(2)) == 1


# ==========================================================================
# INTEGRATION  --  success criteria
# ==========================================================================

class TestIntegration:
    def test_zero_collisions_nominal(self):
        for seed in (1, 2, 3):
            r = Simulation(n_robots=4, seed=seed).run(duration=120)
            assert r["collisions"] == 0, f"seed {seed} collided"

    def test_T13_zero_collisions_under_blackout(self):
        """The differentiator. Safety runs on LiDAR, not on the network, so
        a comms blackout must not produce collisions."""
        sim = Simulation(n_robots=4, seed=3)
        for _ in range(400):
            sim.step()
        sim.comms.cut_everything()
        for _ in range(800):
            sim.step()
        assert sim.metrics.collisions == 0

    def test_robots_degrade_on_comms_loss(self):
        sim = Simulation(n_robots=4, seed=3)
        for _ in range(300):
            sim.step()
        sim.comms.cut_everything()
        for _ in range(300):
            sim.step()
        assert sum(1 for r in sim.robots if r.degraded) >= 3

    def test_tasks_actually_complete(self):
        """Guards the degenerate 'park everything, zero collisions' solution."""
        r = Simulation(n_robots=4, seed=1).run(duration=180)
        assert r["tasks_completed"] >= 3

    def test_robots_stay_on_free_cells(self):
        sim = Simulation(n_robots=4, seed=2)
        for _ in range(900):
            sim.step()
            for rb in sim.robots:
                assert sim.wmap.is_free(*rb.cell), "robot inside structure"

    def test_deterministic_given_seed(self):
        """Benchmarks are meaningless if runs aren't reproducible."""
        a = Simulation(n_robots=4, seed=7).run(duration=60)
        b = Simulation(n_robots=4, seed=7).run(duration=60)
        assert a["tasks_completed"] == b["tasks_completed"]
        assert a["distance_m"] == b["distance_m"]

    def test_T18_beats_stop_and_wait(self):
        """THE SUCCESS CRITERION: >=20% task-time reduction."""
        seeds = [1, 2, 3, 4, 5, 6]
        base, full = [], []
        for s in seeds:
            b = Simulation(n_robots=4, seed=s, congestion=False,
                           speed_adapt=False).run(duration=180)
            f = Simulation(n_robots=4, seed=s, congestion=True,
                           speed_adapt=True).run(duration=180)
            if b["avg_task_time"] > 0 and f["avg_task_time"] > 0:
                base.append(b["avg_task_time"])
                full.append(f["avg_task_time"])
        mb, mf = sum(base) / len(base), sum(full) / len(full)
        gain = 100 * (mb - mf) / mb
        assert gain >= 20.0, f"only {gain:.1f}% improvement (target 20%)"


# ==========================================================================
# LEARNED POLICY
# ==========================================================================

class TestPolicy:
    def test_shapes_and_size(self):
        import numpy as np
        from learned import OBS_DIM, PolicyNet
        net = PolicyNet()
        a, conf = net.predict(np.zeros(OBS_DIM, dtype=np.float32))
        assert 0 <= a < 5 and 0.0 <= conf <= 1.0
        assert net.n_params() < 20000, "too big for edge deployment"

    def test_observation_is_local_only(self):
        """If the policy could see the whole map it would not be
        decentralised. The 7x7 window IS the contribution."""
        import numpy as np
        from learned import OBS_DIM, WINDOW, build_observation
        m, t = WarehouseMap(), ReservationTable()
        obs = build_observation(m, 10, 10, (30, 20), t, 0.0, 1, 0.8, 0.5)
        assert obs.shape == (OBS_DIM,)
        assert OBS_DIM == WINDOW * WINDOW * 2 + 5

    def test_learns_above_chance(self):
        import numpy as np
        from learned import PolicyNet
        rng = np.random.default_rng(0)
        X = rng.normal(size=(400, 103)).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.int64) * 2
        net = PolicyNet()
        net.train(X, y, epochs=30, lr=0.05, verbose=False)
        assert net.accuracy(X, y) > 0.8

    def test_fallback_on_low_confidence(self):
        """Safety: an unsure policy must defer to the deterministic layer."""
        import numpy as np
        from learned import HybridCoordinator, PolicyNet
        hc = HybridCoordinator(PolicyNet(), conf_threshold=0.99)
        _, src = hc.choose(np.zeros(103, dtype=np.float32), rule_action=3)
        assert src == "RULE_LOWCONF"

    def test_no_policy_means_pure_rules(self):
        import numpy as np
        from learned import HybridCoordinator
        hc = HybridCoordinator(None)
        a, src = hc.choose(np.zeros(103, dtype=np.float32), rule_action=2)
        assert a == 2 and src == "RULE"
