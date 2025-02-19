import uuid

class Node:
    def __init__(self, label, properties=None, node_id=None, outgoing_edge_ids=None):
        self.id = node_id or str(uuid.uuid4())
        self.label = label
        self.properties = properties or {}
        self.outgoing_edge_ids = outgoing_edge_ids or []

    def to_dict(self):
        return {
            "id": self.id,
            "label": self.label,
            "properties": self.properties,
            "outgoing_edge_ids": self.outgoing_edge_ids
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            node_id=data["id"],
            label=data["label"],
            properties=data["properties"],
            outgoing_edge_ids=data["outgoing_edge_ids"]
        )

    def __repr__(self):
        return (f"Node(id={self.id}, label={self.label}, "
                f"props={self.properties}, edges={len(self.outgoing_edge_ids)})")


class Edge:
    def __init__(self, label, start_node_id, end_node_id, properties=None, edge_id=None):
        self.id = edge_id or str(uuid.uuid4())
        self.label = label
        self.start_node_id = start_node_id
        self.end_node_id = end_node_id
        self.properties = properties or {}

    def to_dict(self):
        return {
            "id": self.id,
            "label": self.label,
            "start_node_id": self.start_node_id,
            "end_node_id": self.end_node_id,
            "properties": self.properties
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            edge_id=data["id"],
            label=data["label"],
            start_node_id=data["start_node_id"],
            end_node_id=data["end_node_id"],
            properties=data["properties"]
        )

    def __repr__(self):
        return (f"Edge(id={self.id}, label={self.label}, "
                f"start={self.start_node_id}, end={self.end_node_id}, "
                f"props={self.properties})")
