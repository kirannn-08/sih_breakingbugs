"""
Comms mediator  ===  THE ISOLATION BOUNDARY  ===

Every peer-to-peer message passes through here. Nothing else may cross
between robots.

Why this node exists
--------------------
If robots can read each other's state directly (shared object, shared DDS
domain, global variable), your "decentralised" system is quietly reading a
global blackboard and the comms-failure demo proves nothing. This class is
the single choke point that makes the claim testable -- and it gives you a
one-line switch to cut links live on stage.

Models: range limit, packet loss, latency, dead zones, hard link cuts.
"""

from __future__ import annotations

import random
from collections import Counter, deque
from dataclasses import dataclass

from amr_msgs import LinkPacket, BROADCAST


@dataclass
class DeadZone:
    """Rectangular region of poor connectivity (metal racking, RF hole)."""
    x0: float
    y0: float
    x1: float
    y1: float
    deliver_prob: float = 0.1

    def contains(self, x: float, y: float) -> bool:
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1


class CommsMediator:
    def __init__(self, robot_ids: list[int], rng: random.Random | None = None,
                 comm_range: float = 25.0, loss_rate: float = 0.0,
                 latency_mean: float = 0.020, latency_std: float = 0.010):
        self.ids = list(robot_ids)
        self.rng = rng or random.Random(0)
        self.comm_range = comm_range
        self.loss_rate = loss_rate
        self.latency_mean = latency_mean
        self.latency_std = latency_std

        # link_matrix[(i,j)] = False -> hard cut. Demo control.
        self.link_matrix: dict[tuple[int, int], bool] = {
            (i, j): True for i in self.ids for j in self.ids if i != j
        }
        self.dead_zones: list[DeadZone] = []

        self._inflight: list[tuple[float, int, LinkPacket]] = []
        self._inbox: dict[int, deque[LinkPacket]] = {i: deque() for i in self.ids}
        self._positions: dict[int, tuple[float, float]] = {}

        self.sent = 0
        self.delivered = 0
        self.dropped = 0
        self.bytes_sent = 0
        # per-type counters -- the dashboard needs to show that Bid/Claim/
        # Release actually travel, otherwise "decentralised auction" is a
        # claim on a slide rather than something a judge can watch happen.
        self.by_type: Counter = Counter()
        self.bytes_by_type: Counter = Counter()

    # -- demo controls -----------------------------------------------------

    def cut_link(self, a: int, b: int) -> None:
        self.link_matrix[(a, b)] = False
        self.link_matrix[(b, a)] = False

    def restore_link(self, a: int, b: int) -> None:
        self.link_matrix[(a, b)] = True
        self.link_matrix[(b, a)] = True

    def isolate(self, robot_id: int) -> None:
        """Cut ALL links to one robot. Demo scenario 8."""
        for other in self.ids:
            if other != robot_id:
                self.cut_link(robot_id, other)

    def restore_all(self) -> None:
        for k in self.link_matrix:
            self.link_matrix[k] = True

    def cut_everything(self) -> None:
        for k in self.link_matrix:
            self.link_matrix[k] = False

    def link_state(self) -> dict:
        """For the dashboard link-state panel."""
        return {f"{i}->{j}": v for (i, j), v in self.link_matrix.items()}

    # -- position tracking (for range + dead zones) ------------------------

    def update_position(self, robot_id: int, x: float, y: float) -> None:
        self._positions[robot_id] = (x, y)

    # -- transport ---------------------------------------------------------

    def send(self, pkt: LinkPacket, now: float) -> None:
        self.sent += 1
        n = pkt.size_bytes()
        self.bytes_sent += n
        self.by_type[pkt.msg_type] += 1
        self.bytes_by_type[pkt.msg_type] += n
        targets = [i for i in self.ids if i != pkt.src] \
            if pkt.dst == BROADCAST else [pkt.dst]
        for dst in targets:
            if self._deliverable(pkt.src, dst):
                lat = max(0.0, self.rng.gauss(self.latency_mean, self.latency_std))
                self._inflight.append((now + lat, dst, pkt))
            else:
                self.dropped += 1

    def _deliverable(self, src: int, dst: int) -> bool:
        if not self.link_matrix.get((src, dst), False):
            return False
        if self.rng.random() < self.loss_rate:
            return False
        ps, pd = self._positions.get(src), self._positions.get(dst)
        if ps and pd:
            d = ((ps[0] - pd[0]) ** 2 + (ps[1] - pd[1]) ** 2) ** 0.5
            if d > self.comm_range:
                return False
            for dz in self.dead_zones:
                if dz.contains(*ps) or dz.contains(*pd):
                    if self.rng.random() > dz.deliver_prob:
                        return False
        return True

    def step(self, now: float) -> None:
        """Move due packets from in-flight into inboxes."""
        still = []
        for (t_due, dst, pkt) in self._inflight:
            if t_due <= now:
                self._inbox[dst].append(pkt)
                self.delivered += 1
            else:
                still.append((t_due, dst, pkt))
        self._inflight = still

    def receive(self, robot_id: int) -> list[LinkPacket]:
        """Drain one robot's inbox. This is the ONLY way a robot learns
        about peers."""
        box = self._inbox[robot_id]
        out = list(box)
        box.clear()
        return out

    def stats(self) -> dict:
        """
        loss_pct is per DELIVERY ATTEMPT, not per packet.

        A broadcast to N-1 peers is one `sent` but N-1 attempts, so dividing
        drops by `sent` overstated loss by (N-1)x -- it reported 62% for a
        configured 20% at N=4, which flatters the resilience claim.
        """
        attempts = self.delivered + self.dropped
        return {"sent": self.sent, "delivered": self.delivered,
                "dropped": self.dropped, "bytes": self.bytes_sent,
                "attempts": attempts,
                "loss_pct": round(100 * self.dropped / max(1, attempts), 2),
                "by_type": dict(self.by_type),
                "bytes_by_type": dict(self.bytes_by_type)}
