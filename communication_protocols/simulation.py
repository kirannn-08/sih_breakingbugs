import random
import asyncio
import math
from network import CommsMediator, DeadZone
from amr_node import TelemetryExtractor, AMRCommNode
from models import LinkPacket

class SimulationController:
    def __init__(self, num_amrs=5, on_event=None):
        self.num_amrs = num_amrs
        self.robot_ids = list(range(1, num_amrs + 1))
        self.all_nodes = [0] + self.robot_ids # 0 is server
        self.on_event = on_event
        self.sim_time = 0.0
        
        # Grid Graph matches the HTML frontend layout
        self.gridX = [60, 200, 440, 680, 840]
        self.gridY = [220, 410, 630]
        self.nodes = []
        for i, x in enumerate(self.gridX):
            for j, y in enumerate(self.gridY):
                self.nodes.append({'x': float(x), 'y': float(y), 'i': i, 'j': j})
                
        rng = random.Random(42)
        # Using a much larger comm range to match pixel scale (e.g. 300 pixels)
        self.wifi_net = CommsMediator(self.all_nodes, rng, comm_range=300.0, loss_rate=0.01, latency_mean=0.010, name="WIFI", on_event=self._dispatch_event)
        self.wisun_net = CommsMediator(self.all_nodes, rng, comm_range=400.0, loss_rate=0.05, latency_mean=0.080, name="WISUN", on_event=self._dispatch_event)
        
        # Deadzone in pixel coordinates
        self.dz = DeadZone(x0=300, y0=200, x1=700, y1=500)
        self.wifi_net.dead_zones.append(self.dz)
        
        self.extractors = {r_id: TelemetryExtractor(r_id) for r_id in self.robot_ids}
        self.comm_nodes = {r_id: AMRCommNode(r_id, self.wifi_net, self.wisun_net, self.all_nodes, on_event=self._dispatch_event) for r_id in self.all_nodes}
        
        self.positions = {}
        self.current_node = {}
        self.target_node = {}
        self.speeds = {}
        self._init_positions()
        
        self.running = False
        
    def _dispatch_event(self, event_type, packet=None, **kwargs):
        if self.on_event:
            self.on_event(event_type, packet, **kwargs)
            
    def _get_neighbors(self, node):
        neighbors = []
        for n in self.nodes:
            dx = abs(n['i'] - node['i'])
            dy = abs(n['j'] - node['j'])
            if (dx == 1 and dy == 0) or (dx == 0 and dy == 1):
                neighbors.append(n)
        return neighbors
        
    def _pick_random_neighbor(self, node, exclude=None):
        neighbors = self._get_neighbors(node)
        if exclude and len(neighbors) > 1:
            neighbors = [n for n in neighbors if n != exclude]
        return random.choice(neighbors)

    def _init_positions(self):
        # Server pos in pixel coords
        self.positions[0] = [500.0, 40.0]
        self.wifi_net.update_position(0, 500.0, 40.0)
        self.wisun_net.update_position(0, 500.0, 40.0)

        available_nodes = list(self.nodes)
        random.shuffle(available_nodes)

        for i, r_id in enumerate(self.robot_ids):
            node = random.choice(self.nodes)
            node = available_nodes.pop()
            self.current_node[r_id] = node
            self.target_node[r_id] = self._pick_random_neighbor(node)
            self.positions[r_id] = [node['x'], node['y']]
            self.speeds[r_id] = 5.0 + random.uniform(0.0, 2.0)
            
            self.wifi_net.update_position(r_id, *self.positions[r_id])
            self.wisun_net.update_position(r_id, *self.positions[r_id])
            
    def set_dead_zone(self, x0, y0, x1, y1):
        self.dz.bounds = (x0, y0, x1, y1)
        
    def force_dead_zone(self, amr_id, in_dz):
        if amr_id in self.comm_nodes:
            self.comm_nodes[amr_id].update_dead_zone_status(in_dz)
            
    def create_task(self, src_id, dst_id, task_data):
        if src_id in self.comm_nodes:
            self.comm_nodes[src_id].create_task_packet(dst_id, task_data, self.sim_time)
            
    async def step(self, dt=0.05):
        self.sim_time += dt
        
        # 1. Update positions (Waypoint movement + collision avoidance)
        for r_id in self.robot_ids:
            px, py = self.positions[r_id]
            tx = self.target_node[r_id]['x']
            ty = self.target_node[r_id]['y']
            
            dx = tx - px
            dy = ty - py
            dist = math.hypot(dx, dy)
            
            speed = self.speeds[r_id]
            
            can_move = True
            for other_id in self.robot_ids:
                if other_id != r_id:
                    ox, oy = self.positions[other_id]
                    odist = math.hypot(ox - px, oy - py)
                    if 0 < odist < 45 and dist > 0:
                        dot = (dx * (ox - px) + dy * (oy - py)) / (dist * odist)
                        if dot > 0.5: # other is in front (narrower cone to avoid side-by-side jitter)
                            # ALWAYS BOUNCE to prevent pass-throughs and deadlocks
                            temp = self.current_node[r_id]
                            self.current_node[r_id] = self.target_node[r_id]
                            self.target_node[r_id] = temp
                            can_move = False
                                
            if can_move:
                if dist < speed:
                    self.positions[r_id][0] = tx
                    self.positions[r_id][1] = ty
                    prev = self.current_node[r_id]
                    self.current_node[r_id] = self.target_node[r_id]
                    self.target_node[r_id] = self._pick_random_neighbor(self.current_node[r_id], prev)
                else:
                    self.positions[r_id][0] += (dx / dist) * speed
                    self.positions[r_id][1] += (dy / dist) * speed
            
            px, py = self.positions[r_id]
            self.wifi_net.update_position(r_id, px, py)
            self.wisun_net.update_position(r_id, px, py)
            self.extractors[r_id].update_simulated_pose(px, py, linear_v=speed)
            
            self._dispatch_event("AMR_MOVED", amr_id=r_id, x=px, y=py)

            in_dz = self.dz.contains(px, py)
            self.comm_nodes[r_id].update_dead_zone_status(in_dz)
            
        # Dispatch topology
        links = []
        for i in self.all_nodes:
            for j in self.all_nodes:
                if i < j:
                    pi = self.positions[i]
                    pj = self.positions[j]
                    dist = math.hypot(pi[0] - pj[0], pi[1] - pj[1])
                    
                    i_dz = self.dz.contains(*pi) if i != 0 else False
                    j_dz = self.dz.contains(*pj) if j != 0 else False

                    if not i_dz and not j_dz and dist <= self.wifi_net.range:
                        links.append({"src": i, "dst": j, "type": "WIFI"})
                    if dist <= self.wisun_net.range and (i_dz or j_dz): 
                        links.append({"src": i, "dst": j, "type": "WISUN"})
        
        self._dispatch_event("NETWORK_TOPOLOGY", links=links)
        frame_data = {
            "amrs": [{"id": r_id, "x": self.positions[r_id][0], "y": self.positions[r_id][1]} for r_id in self.robot_ids],
            "links": links
        }
        self._dispatch_event("SYNC_FRAME", **frame_data)

        # 2. Extract and broadcast telemetry
        for r_id in self.robot_ids:
            payload = self.extractors[r_id].get_json_payload()
            self.comm_nodes[r_id].broadcast_telemetry(payload, self.sim_time)
            
        # 3. Process queues
        for n_id in self.all_nodes:
            self.comm_nodes[n_id].process_queue(self.sim_time)
            
        # 4. Step networks
        self.wifi_net.step(self.sim_time)
        self.wisun_net.step(self.sim_time)
        
        # 5. Receive packets
        for n_id in self.all_nodes:
            self.comm_nodes[n_id].process_inbox(self.sim_time)

    async def run_loop(self):
        self.running = True
        while self.running:
            await self.step(dt=0.05)
            await asyncio.sleep(0.05)
