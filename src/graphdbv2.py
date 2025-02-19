import json
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict

from kvstorage import KVStorage
from lru_cache import LRUCache
from graph_entities import Node, Edge


class GraphDB:
    """
    A simplified, index-free adjacency graph DB that:
     - uses a KVStorage with batch read/write support,
     - supports multi-threaded JSON (de)serialization for read operations,
     - has LRU caches for nodes/edges.
    """

    NODE_PREFIX = b"N:"
    EDGE_PREFIX = b"E:"

    def __init__(self,
                 storage: KVStorage,
                 cache_capacity=1000,
                 max_workers=4):
        """
        :param storage: KVStorage implementation (LevelDB, LMDB, etc.)
        :param cache_capacity: LRU cache size for nodes and edges
        :param max_workers: number of threads to use for parallel reads/deserialization
        """
        self.storage = storage
        self.node_cache = LRUCache(capacity=cache_capacity)
        self.edge_cache = LRUCache(capacity=cache_capacity)
        self.max_workers = max_workers

    # ------------------------------------------------------------------
    # Key Helpers
    # ------------------------------------------------------------------

    def _make_node_key(self, node_id: str) -> bytes:
        return self.NODE_PREFIX + node_id.encode("utf-8")

    def _make_edge_key(self, edge_id: str) -> bytes:
        return self.EDGE_PREFIX + edge_id.encode("utf-8")

    # ------------------------------------------------------------------
    # CREATE: Single + Batch
    # ------------------------------------------------------------------

    def create_node(self, label, properties=None) -> Node:
        node = Node(label, properties)
        self._put_node(node)
        return node

    def create_edge(self, label, start_node_id, end_node_id, properties=None) -> Edge:
        start_node = self.get_node(start_node_id)
        end_node = self.get_node(end_node_id)
        if not start_node or not end_node:
            raise ValueError("Start or end node does not exist.")

        edge = Edge(label, start_node_id, end_node_id, properties)
        self._put_edge(edge)

        # Update adjacency on the start node
        start_node.outgoing_edge_ids.append(edge.id)
        self._put_node(start_node)
        return edge

    def create_nodes_batch(self, node_specs: List[Dict]) -> List[Node]:
        """
        Create multiple nodes in one batched write operation.
        node_specs is a list of dicts: [{"label": ..., "properties": {...}}, ...]
        """
        nodes = []
        batch_data = {}
        for spec in node_specs:
            node = Node(label=spec.get("label"), properties=spec.get("properties"))
            nodes.append(node)
            key = self._make_node_key(node.id)
            batch_data[key] = json.dumps(node.to_dict()).encode("utf-8")

        # Write them all at once
        self.storage.put_batch(batch_data)

        # Update caches
        for n in nodes:
            self.node_cache.put(n.id, n)

        return nodes

    def create_edges_batch(self, edge_specs: List[Dict]) -> List[Edge]:
        """
        Create multiple edges in one batched operation.
        edge_specs: [
          {
            "label": "FRIEND",
            "start_node_id": "...",
            "end_node_id": "...",
            "properties": {...}
          },
          ...
        ]
        """
        # First, we might want to verify that all start/end nodes exist
        # (This can also be done in parallel if desired.)
        edges = []
        node_updates = {}  # node_id -> updated Node
        edge_batch_data = {}

        for spec in edge_specs:
            label = spec["label"]
            start_id = spec["start_node_id"]
            end_id = spec["end_node_id"]
            props = spec.get("properties", {})

            start_node = self.get_node(start_id)
            end_node = self.get_node(end_id)
            if not start_node or not end_node:
                raise ValueError(f"Start or end node missing: {start_id}, {end_id}")

            e = Edge(label, start_id, end_id, props)
            edges.append(e)

            # Edge KV
            edge_key = self._make_edge_key(e.id)
            edge_batch_data[edge_key] = json.dumps(e.to_dict()).encode("utf-8")

            # Update start node's adjacency
            start_node.outgoing_edge_ids.append(e.id)
            node_updates[start_node.id] = start_node

        # Now we combine edges + updated node data into a single batch
        node_batch_data = {}
        for node in node_updates.values():
            k = self._make_node_key(node.id)
            node_batch_data[k] = json.dumps(node.to_dict()).encode("utf-8")

        # Merge them into one dict
        batch_data = {}
        batch_data.update(edge_batch_data)
        batch_data.update(node_batch_data)

        # Single batch write
        self.storage.put_batch(batch_data)

        # Update caches
        for e in edges:
            self.edge_cache.put(e.id, e)
        for n in node_updates.values():
            self.node_cache.put(n.id, n)

        return edges

    # ------------------------------------------------------------------
    # Internal Put Methods (Single)
    # ------------------------------------------------------------------

    def _put_node(self, node: Node):
        data = json.dumps(node.to_dict()).encode("utf-8")
        self.storage.put(self._make_node_key(node.id), data)
        self.node_cache.put(node.id, node)

    def _put_edge(self, edge: Edge):
        data = json.dumps(edge.to_dict()).encode("utf-8")
        self.storage.put(self._make_edge_key(edge.id), data)
        self.edge_cache.put(edge.id, edge)

    # ------------------------------------------------------------------
    # GET: Single + Batch
    # ------------------------------------------------------------------

    def get_node(self, node_id: str) -> Node or None:
        cached = self.node_cache.get(node_id)
        if cached:
            return cached

        raw = self.storage.get(self._make_node_key(node_id))
        if raw is None:
            return None

        node = self._deserialize_node(raw)
        self.node_cache.put(node_id, node)
        return node

    def get_edge(self, edge_id: str) -> Edge or None:
        cached = self.edge_cache.get(edge_id)
        if cached:
            return cached

        raw = self.storage.get(self._make_edge_key(edge_id))
        if raw is None:
            return None

        edge = self._deserialize_edge(raw)
        self.edge_cache.put(edge_id, edge)
        return edge

    def get_batch_nodes(self, node_ids: List[str]) -> Dict[str, Node]:
        """
        Retrieve multiple nodes in a single batch call, using parallel threads
        for JSON deserialization. Returns {node_id: Node}.
        """
        # Check cache first, collect missed keys
        to_fetch = []
        results = {}
        for nid in node_ids:
            cached = self.node_cache.get(nid)
            if cached:
                results[nid] = cached
            else:
                to_fetch.append(nid)

        if not to_fetch:
            return results  # all were cached

        # Perform a single DB batch get
        raw_map = self.storage.get_batch([self._make_node_key(nid) for nid in to_fetch])

        # We'll do parallel JSON deserialization
        def worker(item):
            """item = (nid, raw_bytes)"""
            nid, raw_bytes = item
            if raw_bytes is None:
                return nid, None
            node = self._deserialize_node(raw_bytes)
            return nid, node

        items = list(raw_map.items())  # [(b"N:...", raw), ...] 
        # Convert key back to node_id
        key_id_map = {}
        for i, (key_bytes, val) in enumerate(items):
            # slice off the prefix b"N:"
            k_str = key_bytes.decode("utf-8")[2:]
            key_id_map[k_str] = val

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = executor.map(worker, key_id_map.items())

        # futures returns an iterator of (nid, node)
        for nid, node in futures:
            if node:
                self.node_cache.put(nid, node)
                results[nid] = node

        return results

    def get_batch_edges(self, edge_ids: List[str]) -> Dict[str, Edge]:
        """
        Similar to get_batch_nodes, but for edges.
        """
        to_fetch = []
        results = {}
        for eid in edge_ids:
            cached = self.edge_cache.get(eid)
            if cached:
                results[eid] = cached
            else:
                to_fetch.append(eid)

        if not to_fetch:
            return results

        raw_map = self.storage.get_batch([self._make_edge_key(eid) for eid in to_fetch])

        def worker(item):
            eid_bytes, raw_bytes = item
            if raw_bytes is None:
                return None
            eid_str = eid_bytes.decode("utf-8")[2:]  # remove prefix b"E:"
            edge = self._deserialize_edge(raw_bytes)
            return (eid_str, edge)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            all_items = list(raw_map.items())  # [(b"E:...", raw), ...]
            futures = executor.map(worker, all_items)

        for res in futures:
            if res:
                (eid, edge) = res
                self.edge_cache.put(eid, edge)
                results[eid] = edge

        return results

    # ------------------------------------------------------------------
    # DESERIALIZATION HELPERS (for parallel JSON)
    # ------------------------------------------------------------------

    def _deserialize_node(self, raw: bytes) -> Node:
        data = json.loads(raw.decode("utf-8"))
        return Node.from_dict(data)

    def _deserialize_edge(self, raw: bytes) -> Edge:
        data = json.loads(raw.decode("utf-8"))
        return Edge.from_dict(data)

    # ------------------------------------------------------------------
    # GRAPH QUERIES
    # ------------------------------------------------------------------

    def get_neighbors(self, node_id: str):
        """
        Return a list of Node objects reachable via outgoing edges.
        Demo uses single-edge fetch but you could do batch fetch as well.
        """
        node = self.get_node(node_id)
        if not node:
            return []

        # We could do a batched get of edges, then a batched get of neighbor nodes
        edge_map = self.get_batch_edges(node.outgoing_edge_ids)
        neighbor_ids = []
        for e in edge_map.values():
            neighbor_ids.append(e.end_node_id)

        # get all neighbor nodes in a batch
        node_map = self.get_batch_nodes(neighbor_ids)
        return list(node_map.values())

    def bfs(self, start_node_id: str, target_label=None):
        """
        Basic BFS. 
        We can also tweak this to do batched fetch of neighbors at each level.
        """
        visited = set()
        queue = deque([start_node_id])
        order = []

        while queue:
            current_id = queue.popleft()
            if current_id in visited:
                continue
            visited.add(current_id)

            node = self.get_node(current_id)
            if not node:
                continue
            order.append(current_id)
            if target_label and node.label == target_label:
                return node

            # BFS next level
            for nbr in self.get_neighbors(current_id):
                if nbr.id not in visited:
                    queue.append(nbr.id)

        if not target_label:
            return order
        return None

    def close(self):
        """Close the underlying storage."""
        self.storage.close()

    def __repr__(self):
        return (f"GraphDB(storage={self.storage}, "
                f"node_cache_size={len(self.node_cache._store)}, "
                f"edge_cache_size={len(self.edge_cache._store)})")
