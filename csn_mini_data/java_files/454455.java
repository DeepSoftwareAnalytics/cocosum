@Beta
  public long length() throws IOException {
    Optional<Long> lengthIfKnown = lengthIfKnown();
    if (lengthIfKnown.isPresent()) {
      return lengthIfKnown.get();
    }

    Closer closer = Closer.create();
    try {
      Reader reader = closer.register(openStream());
      return countBySkipping(reader);
    } catch (Throwable e) {
      throw closer.rethrow(e);
    } finally {
      closer.close();
    }
  }