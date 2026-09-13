import time
from collections import deque
import heapq
from models import MeshPacket, MESH_BROADCAST

class TelemetryExtractor:
    def __init__(self, robot_id: int):
        self.robot_id = robot_id
        self.x = 0.0
        self.y = 0.0
        self.linear_v = 0.0
        self.angular_v = 0.0
        self.current_intent = "IDLE"
        
    def update_simulated_pose(self, x, y, linear_v=0.0, angular_v=0.0):
        self.x = round(x, 2)
        self.y = round(y, 2)
        self.linear_v = round(linear_v, 2)
        self.angular_v = round(angular_v, 2)
        
    def get_json_payload(self) -> dict:
        return {
            "timestamp": time.time(),
            "amr_id": self.robot_id,
            "position": {"x": self.x, "y": self.y},
            "velocity": {"linear": self.linear_v, "angular": self.angular_v},
            "intent": self.current_intent
        }

class AMRCommNode:
    def __init__(self, robot_id: int, wifi_net, wisun_net, all_ids, on_event=None):
        self.robot_id = robot_id
        self.wifi_net = wifi_net
        self.wisun_net = wisun_net
        self.all_ids = all_ids
        self.on_event = on_event
        
        self.peer_interfaces = {r_id: 'WIFI' for r_id in all_ids if r_id != robot_id}
        self.seen_msg_ids = set()
        self.seen_msg_history = deque(maxlen=500)
        
        self.in_dead_zone = False
        self.last_broadcast_time = 0.0
        
        # Queue metrics
        self.queue = [] # Priority queue for outgoing packets
        self.queue_capacity = 100
        self.queue_seq = 0
        
    def update_dead_zone_status(self, is_dead: bool):
        if is_dead != self.in_dead_zone:
            self.in_dead_zone = is_dead
            if self.on_event:
                if is_dead:
                    self.on_event("WIFI_LOST", None, amr_id=self.robot_id)
                    self.on_event("WISUN_ENABLED", None, amr_id=self.robot_id)
                else:
                    self.on_event("WIFI_RECOVERED", None, amr_id=self.robot_id)
                    
    def _is_duplicate(self, msg_id: str) -> bool:
        if msg_id in self.seen_msg_ids:
            return True
        self.seen_msg_ids.add(msg_id)
        self.seen_msg_history.append(msg_id)
        return False
        
    def get_congestion_score(self):
        return len(self.queue) / max(1, self.queue_capacity)

    def queue_packet(self, pkt: MeshPacket):
        if len(self.queue) >= self.queue_capacity:
            if self.on_event:
                self.on_event("PACKET_DROPPED", pkt, reason="QUEUE_FULL", amr_id=self.robot_id)
            return
            
        self.queue_seq += 1
        heapq.heappush(self.queue, (pkt.priority, self.queue_seq, pkt))
        if self.on_event:
            self.on_event("PACKET_QUEUED", pkt, amr_id=self.robot_id, queue_length=len(self.queue))
            if self.get_congestion_score() > 0.8:
                self.on_event("CONGESTION_DETECTED", None, amr_id=self.robot_id, score=self.get_congestion_score())

    def broadcast_telemetry(self, telemetry_payload: dict, now: float):
        interval = 0.5 if self.in_dead_zone else 0.1  # Adaptive: 2 Hz dead zone, 10 Hz Wi-Fi
        if self.get_congestion_score() > 0.5 and not self.in_dead_zone:
            interval = 0.2 # 5 Hz if congested
            
        if now - self.last_broadcast_time < interval:
            return

        self.last_broadcast_time = now
        msg_id = f"TEL_{self.robot_id}_{now:.3f}"
        
        pkt = MeshPacket(
            message_id=msg_id,
            origin_id=self.robot_id,
            forwarder_id=self.robot_id,
            destination_id=MESH_BROADCAST,
            ttl=3,
            hop_count=0,
            message_type="TELEMETRY",
            priority=5, # lowest priority
            payload=telemetry_payload
        )
        self._is_duplicate(msg_id)
        self.queue_packet(pkt)
        if self.on_event:
            self.on_event("PACKET_CREATED", pkt, amr_id=self.robot_id)

    def create_task_packet(self, dst: int, task_data: dict, now: float):
        msg_id = f"TASK_{self.robot_id}_{now:.3f}"
        pkt = MeshPacket(
            message_id=msg_id,
            origin_id=self.robot_id,
            forwarder_id=self.robot_id,
            destination_id=dst,
            ttl=5,
            hop_count=0,
            message_type="TASK",
            priority=1,
            payload=task_data
        )
        self._is_duplicate(msg_id)
        self.queue_packet(pkt)
        if self.on_event:
            self.on_event("TASK_CREATED", pkt, amr_id=self.robot_id)
            self.on_event("PACKET_CREATED", pkt, amr_id=self.robot_id)

    def select_bridge(self, dst_id: int):
        # Find a suitable bridge based on interfaces and congestion
        active_wifi = [r for r, iface in self.peer_interfaces.items() if iface == 'WIFI' and r != self.robot_id]
        if not active_wifi:
            return None
        # Simple static choice for now, can be improved with real congestion metrics if shared
        return random.choice(active_wifi)
        
    def process_queue(self, now: float):
        # Process some packets from queue (rate limiting)
        process_limit = 5
        processed = 0
        while self.queue and processed < process_limit:
            _, _, pkt = heapq.heappop(self.queue)
            
            # Bridge logic / Interface selection
            if self.in_dead_zone:
                pkt.medium = "WISUN"
                self.wisun_net.send(pkt, now)
            else:
                if pkt.destination_id == MESH_BROADCAST:
                    has_wisun_peers = any(v == 'WISUN' for v in self.peer_interfaces.values())
                    if has_wisun_peers and pkt.medium == "WIFI": # Bridge WIFI -> WISUN
                        import copy
                        w_pkt = copy.deepcopy(pkt)
                        w_pkt.medium = "WISUN"
                        self.wisun_net.send(w_pkt, now)
                        if self.on_event:
                            self.on_event("BRIDGE_SELECTED", pkt, amr_id=self.robot_id)
                    pkt.medium = "WIFI"
                    self.wifi_net.send(pkt, now)
                elif pkt.destination_id != MESH_BROADCAST and self.peer_interfaces.get(pkt.destination_id) == 'WISUN':
                    pkt.medium = "WISUN"
                    self.wisun_net.send(pkt, now)
                    if self.on_event:
                        self.on_event("BRIDGE_SELECTED", pkt, amr_id=self.robot_id)
                else:
                    pkt.medium = "WIFI"
                    self.wifi_net.send(pkt, now)
                    
            if self.on_event:
                self.on_event("PACKET_SENT", pkt, amr_id=self.robot_id)
            processed += 1

    def process_inbox(self, now: float):
        wifi_pkts = self.wifi_net.receive(self.robot_id)
        wisun_pkts = self.wisun_net.receive(self.robot_id)
        
        for pkt in wifi_pkts:
            self.peer_interfaces[pkt.origin_id] = 'WIFI'
            self._handle_received_packet(pkt, now)
            
        for pkt in wisun_pkts:
            self.peer_interfaces[pkt.origin_id] = 'WISUN'
            self._handle_received_packet(pkt, now)
            
    def _handle_received_packet(self, pkt: MeshPacket, now: float):
        if self._is_duplicate(pkt.message_id):
            if self.on_event:
                self.on_event("DUPLICATE_DROPPED", pkt, amr_id=self.robot_id)
            return
            
        if pkt.ttl <= 1:
            if self.on_event:
                self.on_event("TTL_EXPIRED", pkt, amr_id=self.robot_id)
            return

        # If it's meant for me or broadcast, process payload
        if pkt.destination_id == self.robot_id or pkt.destination_id == MESH_BROADCAST:
            if pkt.message_type == "TASK":
                if self.on_event:
                    self.on_event("TASK_RECEIVED", pkt, amr_id=self.robot_id)
                # Auto-ACK
                ack_pkt = MeshPacket(
                    message_id=f"ACK_{pkt.message_id}",
                    origin_id=self.robot_id,
                    forwarder_id=self.robot_id,
                    destination_id=pkt.origin_id,
                    ttl=5,
                    hop_count=0,
                    message_type="ACK",
                    priority=2,
                    payload={"ack_for": pkt.message_id, "status": "ACCEPTED"}
                )
                self.queue_packet(ack_pkt)
            elif pkt.message_type == "ACK":
                if self.on_event:
                    self.on_event("ACK_RECEIVED", pkt, amr_id=self.robot_id)

        # Forwarding logic
        if pkt.destination_id != self.robot_id and pkt.destination_id != MESH_BROADCAST:
            import copy
            fwd_pkt = copy.deepcopy(pkt)
            fwd_pkt.ttl -= 1
            fwd_pkt.hop_count += 1
            fwd_pkt.forwarder_id = self.robot_id
            fwd_pkt.route.append(self.robot_id)
            self.queue_packet(fwd_pkt)
            if self.on_event:
                self.on_event("PACKET_FORWARDED", fwd_pkt, amr_id=self.robot_id)
                
        # Bridge logic for TELEMETRY (Wi-SUN -> Wi-Fi)
        elif pkt.destination_id == MESH_BROADCAST and pkt.message_type == "TELEMETRY":
            if not self.in_dead_zone and pkt.medium == "WISUN":
                import copy
                fwd_pkt = copy.deepcopy(pkt)
                fwd_pkt.ttl -= 1
                fwd_pkt.hop_count += 1
                fwd_pkt.forwarder_id = self.robot_id
                fwd_pkt.destination_id = 0 # Send directly to server
                fwd_pkt.route.append(self.robot_id)
                self.queue_packet(fwd_pkt)
                if self.on_event:
                    self.on_event("BRIDGE_SELECTED", fwd_pkt, amr_id=self.robot_id)
                    self.on_event("PACKET_FORWARDED", fwd_pkt, amr_id=self.robot_id)

        # Bridge logic for TASKS (Wi-Fi -> Wi-SUN)
        elif pkt.destination_id == MESH_BROADCAST and pkt.message_type == "TASK":
            if not self.in_dead_zone and pkt.medium == "WIFI":
                active_deadzone_robots = [r_id for r_id, iface in self.peer_interfaces.items() if iface == 'WISUN']
                if active_deadzone_robots:
                    wifi_active_nodes = [r for r, iface in self.peer_interfaces.items() if iface == 'WIFI']
                    wifi_active_nodes.append(self.robot_id)
                    sorted_nodes = sorted(wifi_active_nodes)
                    designated_forwarder = sorted_nodes[int(now) % len(sorted_nodes)]
                    if self.robot_id == designated_forwarder:
                        for dz_robot in active_deadzone_robots:
                            import copy
                            fwd_pkt = copy.deepcopy(pkt)
                            fwd_pkt.ttl -= 1
                            fwd_pkt.hop_count += 1
                            fwd_pkt.forwarder_id = self.robot_id
                            fwd_pkt.destination_id = dz_robot
                            fwd_pkt.route.append(self.robot_id)
                            self.queue_packet(fwd_pkt)
                            if self.on_event:
                                self.on_event("BRIDGE_SELECTED", fwd_pkt, amr_id=self.robot_id)
                                self.on_event("PACKET_FORWARDED", fwd_pkt, amr_id=self.robot_id)

