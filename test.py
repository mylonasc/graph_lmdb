import unittest
import tempfile
import shutil
import os
import pickle

from src.graphdb import GraphDB, ValueSerializer, NoOpSerializer  # Update import paths as needed

class PickleSetSerializer:
    """Simple adjacency serializer using pickle to store sets."""
    def serialize(self, value):
        return pickle.dumps(value)
    def deserialize(self, value):
        return pickle.loads(value)

class TestGraphDB(unittest.TestCase):
    def setUp(self):
        """
        Create a temporary directory for LMDB storage, 
        then initialize the GraphDB within it.
        """
        self.test_dir = tempfile.mkdtemp()
        
        self.db_path = os.path.join(self.test_dir, "test_graph_db")
        
        self.gdb = GraphDB(
            db_path=self.db_path,
            node_features_serializer=ValueSerializer(),
            edge_features_serializer=ValueSerializer(),
            adjacency_serializer=PickleSetSerializer(),
            map_size=10**8  # 100MB (small for testing)
        )
        
    def tearDown(self):
        """
        Close the GraphDB and remove the temporary directory.
        """
        self.gdb.close()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_node_features(self):
        """
        Test that node features can be stored and retrieved properly.
        """
        node_key = "node1"
        features = {"color": "blue", "weight": 1.23}
        
        # Put and get node feature
        self.gdb.put_node_feature(node_key, features)
        retrieved = self.gdb.get_node_feature(node_key)
        
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.get("color"), "blue")
        self.assertEqual(retrieved.get("weight"), 1.23)

    def test_edge_features(self):
        """
        Test that edge features can be stored and retrieved properly.
        """
        edge_key = "edge1"
        features = {"capacity": 100, "length": 42.0}
        
        self.gdb.put_edge_feature(edge_key, features)
        retrieved = self.gdb.get_edge_feature(edge_key)
        
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.get("capacity"), 100)
        self.assertEqual(retrieved.get("length"), 42.0)

    def test_adjacency_basic(self):
        """
        Test that adding an edge connection updates both incoming and outgoing 
        adjacency lists as expected.
        """
        node_a = "A"
        node_b = "B"
        edge_ab = "edge_ab"
        
        # Initially, no adjacency
        self.assertEqual(self.gdb.get_outgoing_edges(node_a), set())
        self.assertEqual(self.gdb.get_incoming_edges(node_b), set())
        
        # Add an edge from A -> B
        self.gdb.add_edge_connection(node_a, node_b, edge_ab)
        
        # Check adjacency
        out_a = self.gdb.get_outgoing_edges(node_a)
        in_b  = self.gdb.get_incoming_edges(node_b)
        
        self.assertIn(edge_ab.encode("utf-8"), out_a)
        self.assertIn(edge_ab.encode("utf-8"), in_b)
        
    def test_adjacency_multiple_edges(self):
        """
        Test that multiple edges can be tracked for each node 
        both as incoming and outgoing edges.
        """
        node1 = "node1"
        node2 = "node2"
        node3 = "node3"
        
        edge1 = "edge1"
        edge2 = "edge2"
        
        # node1 -> node2
        self.gdb.add_edge_connection(node1, node2, edge1)
        # node2 -> node3
        self.gdb.add_edge_connection(node2, node3, edge2)
        
        # Check node1 adjacency
        out_node1 = self.gdb.get_outgoing_edges(node1)
        in_node1  = self.gdb.get_incoming_edges(node1)
        
        self.assertEqual(out_node1, {edge1.encode("utf-8")})
        self.assertEqual(in_node1, set())
        
        # Check node2 adjacency
        out_node2 = self.gdb.get_outgoing_edges(node2)
        in_node2  = self.gdb.get_incoming_edges(node2)
        
        self.assertEqual(out_node2, {edge2.encode("utf-8")})
        self.assertEqual(in_node2, {edge1.encode("utf-8")})
        
        # Check node3 adjacency
        out_node3 = self.gdb.get_outgoing_edges(node3)
        in_node3  = self.gdb.get_incoming_edges(node3)
        
        self.assertEqual(out_node3, set())
        self.assertEqual(in_node3, {edge2.encode("utf-8")})

    def test_non_existent_keys(self):
        """
        Ensure the DB gracefully returns None (or empty set) for 
        node/edge keys that don't exist.
        """
        self.assertIsNone(self.gdb.get_node_feature("does_not_exist"))
        self.assertIsNone(self.gdb.get_edge_feature("no_such_edge"))
        
        self.assertEqual(self.gdb.get_outgoing_edges("no_node"), set())
        self.assertEqual(self.gdb.get_incoming_edges("no_node"), set())

    def test_multiple_updates(self):
        """
        Test that updating features multiple times overwrites old data.
        """
        node_key = "nodeX"
        
        # Put initial feature
        self.gdb.put_node_feature(node_key, {"color": "blue"})
        ret1 = self.gdb.get_node_feature(node_key)
        self.assertEqual(ret1.get("color"), "blue")
        
        # Overwrite with new feature
        self.gdb.put_node_feature(node_key, {"color": "red", "size": 10})
        ret2 = self.gdb.get_node_feature(node_key)
        self.assertEqual(ret2.get("color"), "red")
        self.assertEqual(ret2.get("size"), 10)
        self.assertFalse("blue" in ret2.values())

if __name__ == "__main__":
    unittest.main()
