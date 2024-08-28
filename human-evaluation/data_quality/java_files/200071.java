public void pushMetric(final MetricsRecord mr) {
    lock.lock();
    try {
      intervalHeartBeat();
      try {
        mr.incrMetric(getName(), getPreviousIntervalValue());
      } catch (Exception e) {
        LOG.info("pushMetric failed for " + getName() + "\n" +
            StringUtils.stringifyException(e));
      }
    } finally {
      lock.unlock();
    }
    
  }