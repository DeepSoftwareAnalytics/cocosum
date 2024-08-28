private final boolean parseBoolean(String value)
   {
      return value != null && (value.equalsIgnoreCase("true") || value.equalsIgnoreCase("y") || value.equalsIgnoreCase("yes"));
   }