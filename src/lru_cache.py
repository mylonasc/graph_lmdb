from collections import OrderedDict

class LRUCache:
    """
    Simple LRU cache using OrderedDict.
    """
    def __init__(self, capacity=1000):
        self.capacity = capacity
        self._store = OrderedDict()

    def get(self, key):
        if key not in self._store:
            return None
        value = self._store.pop(key)
        self._store[key] = value  # mark as most recently used
        return value

    def put(self, key, value):
        if key in self._store:
            self._store.pop(key)
        self._store[key] = value
        if len(self._store) > self.capacity:
            self._store.popitem(last=False)  # remove LRU
