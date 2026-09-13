import unittest
import time
from models import LinkPacket, BROADCAST
from network import CommsMediator, DeadZone
from amr_node import AMRCommNode
import random

class TestAMRSimulation(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(42)
        self.all_ids = [1, 2, 3]
        self.wifi_net = CommsMediator(self.all_ids, self.rng, comm_range=60.0)
        self.wisun_net = CommsMediator(self.all_ids, self.rng, comm_range=40.0)
        self.node1 = AMRCommNode(1, self.wifi_net, self.wisun_net, self.all_ids)
        self.node2 = AMRCommNode(2, self.wifi_net, self.wisun_net, self.all_ids)
        self.node3 = AMRCommNode(3, self.wifi_net, self.wisun_net, self.all_ids)
        
        self.wifi_net.update_position(1, 0, 0)
        self.wifi_net.update_position(2, 10, 0)
        self.wifi_net.update_position(3, 20, 0)

        self.wisun_net.update_position(1, 0, 0)
        self.wisun_net.update_position(2, 10, 0)
        self.wisun_net.update_position(3, 20, 0)

    def test_wifi_p2p(self):
        pkt = LinkPacket(
            message_id="TEST_1", origin_id=1, forwarder_id=1, destination_id=2, 
            ttl=3, hop_count=0, message_type="TEST", priority=1
        )
        self.node1.queue_packet(pkt)
        self.node1.process_queue(0.0)
        self.wifi_net.step(0.1)
        self.node2.process_inbox(0.1)
        self.assertTrue(self.node2._is_duplicate("TEST_1"))

    def test_dead_zone_and_wisun_fallback(self):
        dz = DeadZone(15, -5, 25, 5)
        self.wifi_net.dead_zones.append(dz)
        self.node3.update_dead_zone_status(True)
        
        # Node 1 sends to Node 3 via Wi-Fi. Should drop.
        pkt = LinkPacket(
            message_id="TEST_2", origin_id=1, forwarder_id=1, destination_id=3, 
            ttl=3, hop_count=0, message_type="TEST", priority=1
        )
        self.node1.queue_packet(pkt)
        self.node1.process_queue(0.0)
        self.wifi_net.step(0.1)
        self.node3.process_inbox(0.1)
        self.assertFalse(self.node3._is_duplicate("TEST_2")) # dropped
        
        # Now Node 2 (bridge) receives from 1 and bridges to 3 via Wi-SUN
        self.node3.seen_msg_ids.clear()
        self.node2.peer_interfaces[3] = 'WISUN'
        pkt3 = LinkPacket(
            message_id="TEST_3", origin_id=1, forwarder_id=2, destination_id=3, 
            ttl=3, hop_count=1, message_type="TEST", priority=1
        )
        self.node2.queue_packet(pkt3)
        self.node2.process_queue(0.2)
        self.wisun_net.step(0.3)
        self.node3.process_inbox(0.3)
        self.assertTrue(self.node3._is_duplicate("TEST_3"))

    def test_duplicate_suppression(self):
        self.assertFalse(self.node1._is_duplicate("MSG_DUP"))
        self.assertTrue(self.node1._is_duplicate("MSG_DUP"))

    def test_ttl_expiry(self):
        pkt = LinkPacket(
            message_id="TEST_TTL", origin_id=1, forwarder_id=1, destination_id=2, 
            ttl=1, hop_count=0, message_type="TEST", priority=1
        )
        self.node2._handle_received_packet(pkt, 0.0)
        # ttl is 1, handled but dropped before forwarding further
        
if __name__ == '__main__':
    unittest.main()
