class NodeRecord:
    """
    Represents a node in our index-free graph store.
    - node_id: unique integer ID for the node
    - label: optional string label or type (e.g., "Person")
    - first_rel_id: ID of this node's first relationship record (forms a linked list of edges)
    - properties: dict of key-value properties
    """
    __slots__ = ("node_id", "label", "first_rel_id", "properties")

    def __init__(self, node_id, label=None, first_rel_id=None, properties=None):
        self.node_id = node_id
        self.label = label
        self.first_rel_id = first_rel_id  # points to a RelationshipRecord rel_id
        self.properties = properties if properties is not None else {}

    def __repr__(self):
        return f"<NodeRecord id={self.node_id}, label={self.label}, firstRel={self.first_rel_id}, props={self.properties}>"


class RelationshipRecord:
    """
    Represents a relationship (edge).
    - rel_id: unique integer ID for this relationship
    - start_node: ID of the start node
    - end_node: ID of the end node
    - rel_type: string type (e.g., "FRIENDS_WITH")
    - next_rel_for_start: ID of the next relationship for the start node
    - next_rel_for_end: ID of the next relationship for the end node
    - properties: dict of key-value properties
    """
    __slots__ = ("rel_id", "start_node", "end_node", "rel_type",
                 "next_rel_for_start", "next_rel_for_end", "properties")

    def __init__(self, rel_id, start_node, end_node, rel_type=None,
                 next_rel_for_start=None, next_rel_for_end=None, properties=None):
        self.rel_id = rel_id
        self.start_node = start_node
        self.end_node = end_node
        self.rel_type = rel_type
        self.next_rel_for_start = next_rel_for_start
        self.next_rel_for_end = next_rel_for_end
        self.properties = properties if properties is not None else {}

    def __repr__(self):
        return (f"<RelationshipRecord id={self.rel_id}, "
                f"({self.start_node})-[{self.rel_type}]->({self.end_node}), "
                f"nextForStart={self.next_rel_for_start}, "
                f"nextForEnd={self.next_rel_for_end}, props={self.properties}>")


class IndexFreeGraphDB:
    """
    An in-memory, index-free adjacency graph DB.
    Stores:
      - node_records: dict[node_id -> NodeRecord]
      - rel_records: dict[rel_id -> RelationshipRecord]

    Provides:
      - create_node()
      - create_relationship()
      - get_node()
      - get_relationship()
      - get_neighbors(node_id)
      - simple traversal example
    """
    def __init__(self):
        self.node_records = {}  # node_id -> NodeRecord
        self.rel_records = {}   # rel_id -> RelationshipRecord

        self._next_node_id = 1
        self._next_rel_id = 1

    def create_node(self, label=None, properties=None):
        """
        Creates a new node record with an automatically assigned ID.
        Returns the node_id.
        """
        node_id = self._next_node_id
        self._next_node_id += 1

        node = NodeRecord(
            node_id=node_id,
            label=label,
            first_rel_id=None,
            properties=properties
        )
        self.node_records[node_id] = node
        return node_id

    def create_relationship(self, start_node_id, end_node_id, rel_type=None, properties=None):
        """
        Creates a new relationship record connecting (start_node_id) -> (end_node_id).
        - We insert the relationship record
        - We update the adjacency links:
           * node_records[start_node_id].first_rel_id -> this new relationship (or chain it)
           * node_records[end_node_id].first_rel_id -> possibly chain it if needed
           * relationship.next_rel_for_start -> old 'first_rel_id' for the start node
           * relationship.next_rel_for_end -> old 'first_rel_id' for the end node
        Returns the rel_id.
        """
        if start_node_id not in self.node_records or end_node_id not in self.node_records:
            raise ValueError("Invalid start or end node ID.")

        rel_id = self._next_rel_id
        self._next_rel_id += 1

        start_node_record = self.node_records[start_node_id]
        end_node_record = self.node_records[end_node_id]

        # The new relationship's "nextRelForStart" should point to the existing firstRel of start node
        next_rel_for_start = start_node_record.first_rel_id
        # The new relationship's "nextRelForEnd" should point to the existing firstRel of end node
        next_rel_for_end = end_node_record.first_rel_id

        # Create the relationship record
        rel_record = RelationshipRecord(
            rel_id=rel_id,
            start_node=start_node_id,
            end_node=end_node_id,
            rel_type=rel_type,
            next_rel_for_start=next_rel_for_start,
            next_rel_for_end=next_rel_for_end,
            properties=properties
        )
        self.rel_records[rel_id] = rel_record

        # Update the start node's first_rel_id to point to this new relationship
        start_node_record.first_rel_id = rel_id
        # Update the end node's first_rel_id to point to this new relationship
        end_node_record.first_rel_id = rel_id

        return rel_id

    def get_node(self, node_id):
        """Returns the NodeRecord for the given node_id (or None if not found)."""
        return self.node_records.get(node_id)

    def get_relationship(self, rel_id):
        """Returns the RelationshipRecord for the given rel_id (or None if not found)."""
        return self.rel_records.get(rel_id)

    def get_neighbors(self, node_id):
        """
        Returns a list of (neighbor_node_id, rel_id) pairs for all relationships
        emanating from the given node. (Assumes undirected or out+in edges are relevant.)
        Follows the index-free linked list of relationship records.
        """
        if node_id not in self.node_records:
            return []

        node_record = self.node_records[node_id]
        neighbors = []
        visited_rels = set()  # to avoid duplicates or loops in the linked list

        # We'll walk the chain of relationships from the node's first_rel_id
        current_rel_id = node_record.first_rel_id
        while current_rel_id is not None and current_rel_id not in visited_rels:
            visited_rels.add(current_rel_id)

            rel_record = self.rel_records[current_rel_id]
            if rel_record is None:
                break

            # Determine the "other" side of the relationship
            if rel_record.start_node == node_id:
                neighbor_id = rel_record.end_node
                # Next relationship for the start side
                next_rel_id = rel_record.next_rel_for_start
            else:
                neighbor_id = rel_record.start_node
                # Next relationship for the end side
                next_rel_id = rel_record.next_rel_for_end

            neighbors.append((neighbor_id, current_rel_id))
            current_rel_id = next_rel_id

        return neighbors

    def traverse(self, start_node_id, depth=1):
        """
        Perform a simple BFS-like traversal up to `depth` hops from `start_node_id`.
        Returns a dict mapping each node to the distance from start_node_id.
        """
        from collections import deque

        if start_node_id not in self.node_records:
            return {}

        queue = deque([(start_node_id, 0)])
        visited = {start_node_id: 0}

        while queue:
            current_node, dist = queue.popleft()
            if dist >= depth:
                # We stop if we've reached the desired depth
                continue

            # Explore neighbors
            for (nbr_id, rel_id) in self.get_neighbors(current_node):
                if nbr_id not in visited:
                    visited[nbr_id] = dist + 1
                    queue.append((nbr_id, dist + 1))

        return visited

    def __repr__(self):
        return (f"IndexFreeGraphDB(\n"
                f"  Nodes={self.node_records},\n"
                f"  Relationships={self.rel_records}\n"
                f")")


if __name__ == "__main__":
    # Example usage:

    db = IndexFreeGraphDB()

    # Create nodes
    alice_id = db.create_node(label="Person", properties={"name": "Alice"})
    bob_id = db.create_node(label="Person", properties={"name": "Bob"})
    charlie_id = db.create_node(label="Person", properties={"name": "Charlie"})

    # Create relationships
    r1_id = db.create_relationship(alice_id, bob_id, rel_type="FRIENDS_WITH", properties={"since": "2023-01-05"})
    r2_id = db.create_relationship(bob_id, charlie_id, rel_type="FRIENDS_WITH", properties={"since": "2023-01-06"})

    # Print out the DB
    print(db)

    # Get neighbors of Alice
    alice_neighbors = db.get_neighbors(alice_id)
    print(f"Alice's neighbors: {alice_neighbors}")  
    # e.g. [ (bob_id, r1_id) ]

    # Depth-2 traversal from Alice
    visited = db.traverse(alice_id, depth=2)
    print(f"Traversal from Alice (node {alice_id}, depth=2) -> {visited}")