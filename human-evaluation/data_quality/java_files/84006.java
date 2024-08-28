public static ProtoFileElement parse(Location location, String data) {
    return new ProtoParser(location, data.toCharArray()).readProtoFile();
  }