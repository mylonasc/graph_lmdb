# example_usage_leveldb.py
from graphdbv2 import GraphDB
from kvstorage import LevelDBStorage

def main():
    storage = LevelDBStorage(db_path="./my_leveldb_graph", create_if_missing=True)
    graph = GraphDB(storage=storage, cache_capacity=1000, max_workers=4)

    # --- Batch create nodes ---
    node_specs = [
        {"label": "Person", "properties": {"name": "Alice"}},
        {"label": "Person", "properties": {"name": "Bob"}},
        {"label": "Drink",  "properties": {"flavor": "Coffee"}},
    ]
    created_nodes = graph.create_nodes_batch(node_specs)
    alice, bob, coffee = created_nodes  # for convenience

    # --- Batch create edges ---
    edge_specs = [
        {"label": "FRIEND", "start_node_id": alice.id, "end_node_id": bob.id},
        {"label": "LIKES",  "start_node_id": alice.id, "end_node_id": coffee.id},
    ]
    created_edges = graph.create_edges_batch(edge_specs)

    # --- Query neighbors of Alice ---
    neighbors = graph.get_neighbors(alice.id)
    print("Neighbors of Alice:", neighbors)

    # --- BFS for a 'Drink' node from Alice ---
    found_drink = graph.bfs(start_node_id=alice.id, target_label="Drink")
    print("BFS found a Drink from Alice:", found_drink)

    graph.close()

if __name__ == "__main__":
    main()
