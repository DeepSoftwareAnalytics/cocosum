public static ResponseField forInt(String responseName, String fieldName, Map<String, Object> arguments,
      boolean optional, List<Condition> conditions) {
    return new ResponseField(Type.INT, responseName, fieldName, arguments, optional, conditions);
  }