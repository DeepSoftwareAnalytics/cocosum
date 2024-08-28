public static RubyRange<Long> range(long start, long end) {
    return new RubyRange<>(LongSuccessor.getInstance(), start, end,
        Interval.CLOSED);
  }