
import os
import json
import uuid
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

# Third-party libraries for LevelDB and LMDB
try:
    import plyvel
    HAS_PLYVEL = True
except ImportError:
    HAS_PLYVEL = False

try:
    import lmdb
    HAS_LMDB = True
except ImportError:
    HAS_LMDB = False


class KVStorage(ABC):
    """
    An abstract interface for key-value storage.
    Now includes batch read/write methods.
    """

    @abstractmethod
    def put(self, key: bytes, value: bytes) -> None:
        """Store key-value pair in the DB."""
        pass

    @abstractmethod
    def put_batch(self, items: Dict[bytes, bytes]) -> None:
        """Store multiple key-value pairs atomically (if supported)."""
        pass

    @abstractmethod
    def get(self, key: bytes) -> Optional[bytes]:
        """Retrieve the value for 'key' if it exists, else None."""
        pass

    @abstractmethod
    def get_batch(self, keys: List[bytes]) -> Dict[bytes, Optional[bytes]]:
        """
        Retrieve multiple keys at once.
        Return a dict mapping each key -> its value or None if missing.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the storage."""
        pass


class LevelDBStorage(KVStorage):
    """
    A LevelDB-based implementation using plyvel.
    Demonstrates batched writes via write_batch().
    Batched reads are simulated by iterating gets (or parallelizing).
    """
    def __init__(self, db_path: str, create_if_missing=True):
        if not HAS_PLYVEL:
            raise ImportError("plyvel is not installed. Please install it for LevelDB support.")

        self._db = plyvel.DB(db_path, create_if_missing=create_if_missing)

    def put(self, key: bytes, value: bytes) -> None:
        self._db.put(key, value)

    def put_batch(self, items: Dict[bytes, bytes]) -> None:
        with self._db.write_batch() as wb:
            for k, v in items.items():
                wb.put(k, v)

    def get(self, key: bytes) -> Optional[bytes]:
        return self._db.get(key)

    def get_batch(self, keys: List[bytes]) -> Dict[bytes, Optional[bytes]]:
        # LevelDB does not have a "multi-get" call, so we do them individually.
        # In a real system, you might do this in parallel with a thread pool.
        results = {}
        for k in keys:
            results[k] = self._db.get(k)
        return results

    def close(self) -> None:
        self._db.close()


class LMDBStorage(KVStorage):
    """
    An LMDB-based implementation using python-lmdb.
    Demonstrates batched writes in a single transaction,
    and a multi-get approach in a read transaction.
    """
    def __init__(self, db_path: str, map_size=1024 * 1024 * 100, **kwargs):
        """
        :param db_path: directory path for LMDB environment
        :param map_size: max size of the database in bytes
        """
        if not HAS_LMDB:
            raise ImportError("python-lmdb is not installed. Please install it for LMDB support.")

        os.makedirs(db_path, exist_ok=True)
        self._env = lmdb.open(db_path, map_size=map_size, **kwargs)

    def put(self, key: bytes, value: bytes) -> None:
        with self._env.begin(write=True) as txn:
            txn.put(key, value)

    def put_batch(self, items: Dict[bytes, bytes]) -> None:
        with self._env.begin(write=True) as txn:
            for k, v in items.items():
                txn.put(k, v)

    def get(self, key: bytes) -> Optional[bytes]:
        with self._env.begin(write=False) as txn:
            return txn.get(key)

    def get_batch(self, keys: List[bytes]) -> Dict[bytes, Optional[bytes]]:
        results = {}
        with self._env.begin(write=False) as txn:
            for k in keys:
                results[k] = txn.get(k)
        return results

    def close(self) -> None:
        self._env.close()
