public void remove(String correlationId, String key) {
		synchronized (_lock) {
			// Get the entry
			CacheEntry entry = _cache.get(key);

			// Remove entry from the cache
			if (entry != null) {
				_cache.remove(key);
				_count--;
			}
		}
	}