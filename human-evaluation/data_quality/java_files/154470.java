public synchronized Value borrow(Key key) throws SQLException {
    Value value = cache.remove(key);
    if (value == null) {
      return createAction.create(key);
    }
    currentSize -= value.getSize();
    return value;
  }