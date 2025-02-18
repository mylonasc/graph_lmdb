import lmdb
import pickle
from collections import deque

class ValueSerializer:
    """
    A default interface for value serialization. In many real-world cases, 
    you'll want more advanced or different serialization (e.g. JSON, MsgPack, etc.).
    """
    def serialize(self, value):
        return pickle.dumps(value)
    
    def deserialize(self, value):
        return pickle.loads(value)


class NoOpSerializer:
    """
    A serializer that does nothing for cases (like adjacency lists) 
    where you want to store raw bytes or some other custom format.
    """
    def serialize(self, value):
        return value  # Expecting 'value' is already bytes
    
    def deserialize(self, value):
        return value  # Just return the raw bytes


class GraphDB:
    def __init__(
        self, 
        db_path, 
        node_features_serializer=None, 
        edge_features_serializer=None,
        adjacency_serializer=None,
        map_size=10**9,
        **lmdb_kwargs
    ):
        """
        :param db_path: Path to the LMDB environment directory (or file).
        :param node_features_serializer: Serializer for node feature values.
        :param edge_features_serializer: Serializer for edge feature values.
        :param adjacency_serializer: Serializer for adjacency lists.
        :param map_size: Maximum size for the LMDB environment (in bytes).
        :param lmdb_kwargs: Additional keyword args for lmdb.open().
        """
        
        # Default serializers if not provided
        self.node_features_serializer = node_features_serializer or ValueSerializer()
        self.edge_features_serializer = edge_features_serializer or ValueSerializer()
        self.adjacency_serializer = adjacency_serializer or NoOpSerializer()
        
        # Open the LMDB environment with support for multiple sub-dbs
        self.env = lmdb.open(
            db_path,
            max_dbs=4,
            map_size=map_size,
            **lmdb_kwargs
        )
        
        # Create sub-databases:
        with self.env.begin(write=True) as txn:
            self.node_features_db = self.env.open_db(b'node_features', txn=txn)
            self.edge_features_db = self.env.open_db(b'edge_features', txn=txn)
            self.from_db         = self.env.open_db(b'from_adjacency', txn=txn)
            self.to_db           = self.env.open_db(b'to_adjacency', txn=txn)
    
    #
    #  Node Features
    #
    
    def put_node_feature(self, node_key, features):
        """
        Insert or update the feature record for a node.
        
        :param node_key: The identifier for the node (string, int, etc.).
        :param features: Arbitrary structure representing the node’s features.
        """
        serialized_value = self.node_features_serializer.serialize(features)
        
        with self.env.begin(write=True, db=self.node_features_db) as txn:
            txn.put(self._to_bytes(node_key), serialized_value)
    
    def get_node_feature(self, node_key):
        """
        Retrieve the feature record for a node.
        
        :param node_key: The identifier for the node.
        :return: Deserialized node feature dict, or None if not found.
        """
        with self.env.begin(write=False, db=self.node_features_db) as txn:
            raw_val = txn.get(self._to_bytes(node_key))
            if raw_val is None:
                return None
            return self.node_features_serializer.deserialize(raw_val)
    
    #
    #  Edge Features
    #
    
    def put_edge_feature(self, edge_key, features):
        """
        Insert or update the feature record for an edge.
        
        :param edge_key: The identifier for the edge.
        :param features: Arbitrary structure representing the edge’s features.
        """
        serialized_value = self.edge_features_serializer.serialize(features)
        
        with self.env.begin(write=True, db=self.edge_features_db) as txn:
            txn.put(self._to_bytes(edge_key), serialized_value)
    
    def get_edge_feature(self, edge_key):
        """
        Retrieve the feature record for an edge.
        
        :param edge_key: The identifier for the edge.
        :return: Deserialized edge feature dict, or None if not found.
        """
        with self.env.begin(write=False, db=self.edge_features_db) as txn:
            raw_val = txn.get(self._to_bytes(edge_key))
            if raw_val is None:
                return None
            return self.edge_features_serializer.deserialize(raw_val)
    
    #
    # Adjacency management
    #
    
    def add_edge_connection(self, from_node, to_node, edge_key):
        """
        Connect an edge between `from_node` and `to_node`.
        This means we must update two adjacency sub-databases:
          - from_db (from_node -> list of edges)
          - to_db   (to_node   -> list of edges)
        
        :param from_node: The node key acting as "source".
        :param to_node:   The node key acting as "destination".
        :param edge_key:  The edge identifier.
        """
        from_node_b = self._to_bytes(from_node)
        to_node_b   = self._to_bytes(to_node)
        edge_key_b  = self._to_bytes(edge_key)
        
        with self.env.begin(write=True) as txn:
            # Update from_db
            from_edges = self._get_edges_txn(txn, from_node_b, self.from_db)
            from_edges.add(edge_key_b)
            self._put_edges_txn(txn, from_node_b, from_edges, self.from_db)
            
            # Update to_db
            to_edges = self._get_edges_txn(txn, to_node_b, self.to_db)
            to_edges.add(edge_key_b)
            self._put_edges_txn(txn, to_node_b, to_edges, self.to_db)
    
    def get_outgoing_edges(self, node_key):
        """
        Get all edges from the perspective of `node_key` as a 'from_node'.
        
        :param node_key: The node key.
        :return: A set of edge keys (as bytes or deserialized), 
                 depending on how you handle adjacency serialization.
        """
        with self.env.begin(write=False, db=self.from_db) as txn:
            node_key_b = self._to_bytes(node_key)
            raw_val = txn.get(node_key_b)
            if raw_val is None:
                return set()
            edges = self.adjacency_serializer.deserialize(raw_val)
            # If we stored as bytes, `edges` might be a set() or list() of bytes
            return edges
    
    def get_incoming_edges(self, node_key):
        """
        Get all edges from the perspective of `node_key` as a 'to_node'.
        
        :param node_key: The node key.
        :return: A set of edge keys.
        """
        with self.env.begin(write=False, db=self.to_db) as txn:
            node_key_b = self._to_bytes(node_key)
            raw_val = txn.get(node_key_b)
            if raw_val is None:
                return set()
            edges = self.adjacency_serializer.deserialize(raw_val)
            return edges
    
    #
    #  Helper methods
    #
    
    def _to_bytes(self, key):
        """Convert a string/int/etc. to bytes as the LMDB key."""
        if isinstance(key, str):
            return key.encode('utf-8')
        elif isinstance(key, int):
            return str(key).encode('utf-8')
        elif isinstance(key, bytes):
            return key
        else:
            # Convert arbitrary object to string, then to bytes
            return str(key).encode('utf-8')
    
    def _get_edges_txn(self, txn, node_key_b, dbi):
        """
        Retrieves the adjacency list/set for node_key_b within transaction `txn`.
        Returns a Python set of edge keys (bytes).
        """
        raw_val = txn.get(node_key_b, db=dbi)
        if raw_val is None:
            return set()
        return self.adjacency_serializer.deserialize(raw_val)
    
    def _put_edges_txn(self, txn, node_key_b, edges_set, dbi):
        """
        Stores the adjacency set for node_key_b as serialized bytes.
        """
        raw_val = self.adjacency_serializer.serialize(edges_set)
        txn.put(node_key_b, raw_val, db=dbi)
    
    def close(self):
        """Close the LMDB environment (final cleanup)."""
        self.env.close()
    
    def conditional_bfs(self, start_node, edge_condition, max_levels=3):
        """
        Perform a BFS starting from `start_node`, but only traverse edges
        where `edge_condition(edge_feature)` is True.
        
        :param start_node: The node key from which to start the BFS.
        :param edge_condition: A function(edge_feature: dict) -> bool
                              that returns True if we should traverse
                              the edge, and False otherwise.
        :param max_levels: Maximum depth (level) to explore in the BFS.
        
        :return: A dict mapping level -> list of nodes at that level. E.g.:
                 {
                   0: [start_node],
                   1: [...],
                   2: [...],
                   3: [...]
                 }
        """
        visited = set()
        queue = deque([(start_node, 0)])
        
        # We'll store nodes by level in a dict
        levels = {level: [] for level in range(max_levels + 1)}
        
        while queue:
            node, depth = queue.popleft()
            
            # Skip if we've already visited
            if node in visited:
                continue
            
            # Mark this node as visited
            visited.add(node)
            
            # Record node in the appropriate level (if within max_levels)
            if depth <= max_levels:
                levels[depth].append(node)
            
            # If we're not at the max depth, explore further
            if depth < max_levels:
                # Get all outgoing edges from this node
                out_edges = self.get_outgoing_edges(node)
                
                for edge_key_bytes in out_edges:
                    edge_key_str = edge_key_bytes.decode("utf-8")
                    
                    # Retrieve the edge feature (which presumably contains "from" and "to", or "type")
                    edge_feat = self.get_edge_feature(edge_key_str)
                    if edge_feat is None:
                        continue  # Edge feature not found (shouldn't happen if well-formed)
                    
                    # Check user-defined condition on edge features
                    if edge_condition(edge_feat):
                        # If condition passes, proceed to the 'to' node
                        neighbor = edge_feat.get("to")
                        if neighbor not in visited:
                            queue.append((neighbor, depth + 1))
        
        return levels

#
# Example usage
#
if __name__ == "__main__":
    # A simple adjacency serializer that stores sets of bytes
    # via pickle (or you could store them raw if you prefer).
    class PickleSetSerializer:
        def serialize(self, value):
            return pickle.dumps(value)
        def deserialize(self, value):
            return pickle.loads(value)
    
    # Create graph DB
    gdb = GraphDB(
        db_path="my_graph_db",
        node_features_serializer=ValueSerializer(),
        edge_features_serializer=ValueSerializer(),
        adjacency_serializer=PickleSetSerializer(),
        map_size=1_000_000_000,  # 1GB
    )
    
    # Insert some node features
    gdb.put_node_feature("node1", {"color": "blue", "weight": 1.23})
    gdb.put_node_feature("node2", {"color": "red",  "weight": 2.34})
    
    # Insert some edge features
    gdb.put_edge_feature("edge1", {"capacity": 10})
    gdb.put_edge_feature("edge2", {"capacity": 20})
    
    # Connect edges
    gdb.add_edge_connection("node1", "node2", "edge1")
    gdb.add_edge_connection("node2", "node1", "edge2")
    
    # Retrieve node features
    node1_feats = gdb.get_node_feature("node1")
    print("Node1 features:", node1_feats)
    
    # Retrieve edge features
    edge1_feats = gdb.get_edge_feature("edge1")
    print("Edge1 features:", edge1_feats)
    
    # Retrieve adjacency
    outgoing_from_node1 = gdb.get_outgoing_edges("node1")
    print("Node1 outgoing edges:", outgoing_from_node1)
    
    incoming_to_node2 = gdb.get_incoming_edges("node2")
    print("Node2 incoming edges:", incoming_to_node2)
    
    # Clean up
    gdb.close()
