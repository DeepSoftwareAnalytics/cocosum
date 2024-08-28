public boolean isEmpty() throws IOException {
    Optional<Long> lengthIfKnown = lengthIfKnown();
    if (lengthIfKnown.isPresent()) {
      return lengthIfKnown.get() == 0L;
    }
    Closer closer = Closer.create();
    try {
      Reader reader = closer.register(openStream());
      return reader.read() == -1;
    } catch (Throwable e) {
      throw closer.rethrow(e);
    } finally {
      closer.close();
    }
  }