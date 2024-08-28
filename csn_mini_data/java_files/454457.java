public String read() throws IOException {
    Closer closer = Closer.create();
    try {
      Reader reader = closer.register(openStream());
      return CharStreams.toString(reader);
    } catch (Throwable e) {
      throw closer.rethrow(e);
    } finally {
      closer.close();
    }
  }