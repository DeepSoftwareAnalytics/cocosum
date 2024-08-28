public long getLong(String key, long default_) {
    Object o = get(key);
    return o instanceof Number ? ((Number) o).longValue() : default_;
  }