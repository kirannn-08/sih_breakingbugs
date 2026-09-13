import json
from dataclasses import dataclass, field
import time

MESH_BROADCAST = -1

@dataclass
class MeshPacket:
    message_id: str
    origin_id: int
    forwarder_id: int
    destination_id: int
    ttl: int
    hop_count: int
    message_type: str
    priority: int
    medium: str = "WIFI"
    payload: dict = field(default_factory=dict)
    status: str = "CREATED"
    route: list = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def size_bytes(self) -> int:
        try:
            return len(json.dumps({
                "message_id": self.message_id,
                "origin_id": self.origin_id,
                "forwarder_id": self.forwarder_id,
                "destination_id": self.destination_id,
                "ttl": self.ttl,
                "hop_count": self.hop_count,
                "message_type": self.message_type,
                "priority": self.priority,
                "medium": self.medium,
                "payload": self.payload
            }).encode('utf-8'))
        except Exception:
            return 128
