import math
import heapq
import random
from models import MeshPacket, MESH_BROADCAST

class RFDeadZone:
    def __init__(self, x0, y0, x1, y1):
        self.bounds = (x0, y0, x1, y1)
        
    def contains(self, x, y):
        x0, y0, x1, y1 = self.bounds
        return x0 <= x <= x1 and y0 <= y <= y1

class RadioMedium:
    """Mock Network Physics Engine"""
    def __init__(self, robot_ids, rng, comm_range, loss_rate=0.0, latency_mean=0.0, name="NET", on_event=None):
        self.name = name
        self.positions = {}
        self.inboxes = {r: [] for r in robot_ids}
        self.dead_zones = []
        self.range = comm_range
        self.loss_rate = loss_rate
        self.latency_mean = latency_mean
        self.rng = rng
        self.flight_queue = [] 
        self.seq = 0
        self.on_event = on_event
        
    def update_position(self, r_id, x, y):
        self.positions[r_id] = (x, y)
        if r_id not in self.inboxes:
            self.inboxes[r_id] = []
        
    def is_in_dead_zone(self, x, y):
        return any(dz.contains(x, y) for dz in self.dead_zones)
        
    def send(self, pkt: MeshPacket, now: float):
        src_pos = self.positions.get(pkt.forwarder_id)
        if not src_pos:
            return
            
        if self.dead_zones and self.is_in_dead_zone(*src_pos):
            if self.on_event:
                self.on_event("PACKET_DROPPED", pkt, reason="DEADZONE_SENDER")
            return
        if self.on_event and pkt.status != "DELIVERING":
            self.on_event("PACKET_TRANSMITTING", pkt)
            
        for r_id, pos in self.positions.items():
            if r_id == pkt.forwarder_id:
                continue
                
            if pkt.destination_id != MESH_BROADCAST and pkt.destination_id != r_id:
                continue
                
            if self.dead_zones and self.is_in_dead_zone(*pos):
                if self.on_event:
                    self.on_event("PACKET_DROPPED", pkt, reason="DEADZONE_RECEIVER")
                continue
                
            dist = math.hypot(pos[0] - src_pos[0], pos[1] - src_pos[1])
            if dist > self.range:
                if self.on_event:
                    self.on_event("PACKET_DROPPED", pkt, reason="OUT_OF_RANGE")
                continue
                
            if self.rng.random() <= self.loss_rate:
                if self.on_event:
                    self.on_event("PACKET_DROPPED", pkt, reason="RF_LOSS")
                continue
                
            delay = 0.001
            if self.latency_mean > 0.0:
                jitter = self.rng.gauss(0, self.latency_mean * 0.15)
                delay = max(0.001, self.latency_mean + jitter)
                
            delivery_time = now + delay
            self.seq += 1
            
            # Make a copy for delivery to this specific node
            import copy
            delivered_pkt = copy.deepcopy(pkt)
            delivered_pkt.status = "DELIVERING"
            
            heapq.heappush(self.flight_queue, (delivery_time, self.seq, r_id, delivered_pkt))

    def receive(self, r_id):
        pkts = self.inboxes.get(r_id, [])
        self.inboxes[r_id] = []
        return pkts
        
    def step(self, now):
        while self.flight_queue and self.flight_queue[0][0] <= now:
            _, _, r_id, pkt = heapq.heappop(self.flight_queue)
            if r_id in self.inboxes:
                pkt.status = "DELIVERED"
                self.inboxes[r_id].append(pkt)
                if self.on_event:
                    self.on_event("PACKET_DELIVERED", pkt, receiver_id=r_id)
