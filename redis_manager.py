import redis
import pickle
import uuid
import time
from typing import Any, Optional
from config import RedisConfig

class RedisManager:
    def __init__(self, config: RedisConfig):
        self.client = redis.Redis(host=config.host, port=config.port)
        self.max_memory_gb = config.max_memory_gb

    def wait_for_memory(self, timeout: int = 300) -> bool:
        """Wait until Redis has enough memory available."""
        start_time = time.time()
        while True:
            if (self.client.info('memory')['used_memory']/1073741824) <= self.max_memory_gb:
                return True
            if time.time() - start_time > timeout:
                return False
            time.sleep(5)

    def store_data(self, data: Any) -> Optional[str]:
        """Store data in Redis and return the key."""
        if not self.wait_for_memory():
            return None
            
        key = str(uuid.uuid4())
        try:
            self.client.set(key, pickle.dumps(data))
            return key
        except Exception as e:
            print(f"Error storing data in Redis: {e}")
            return None

    def get_data(self, key: str) -> Optional[Any]:
        """Retrieve data from Redis by key."""
        try:
            data = self.client.get(key)
            if data:
                return pickle.loads(data)
            return None
        except Exception as e:
            print(f"Error retrieving data from Redis: {e}")
            return None

    def delete_data(self, key: str) -> bool:
        """Delete data from Redis by key."""
        try:
            return bool(self.client.delete(key))
        except Exception as e:
            print(f"Error deleting data from Redis: {e}")
            return False

    def get_all_keys(self) -> list:
        """Get all keys from Redis."""
        try:
            return self.client.keys()
        except Exception as e:
            print(f"Error getting keys from Redis: {e}")
            return []

    def get_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        try:
            return self.client.info('memory')['used_memory']/1073741824
        except Exception as e:
            print(f"Error getting memory usage: {e}")
            return 0.0 