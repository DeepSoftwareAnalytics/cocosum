@Override
  public final String escape(String s) {
    checkNotNull(s); // GWT specific check (do not optimize)
    for (int i = 0; i < s.length(); i++) {
      char c = s.charAt(i);
      if ((c < replacementsLength && replacements[c] != null)
          || c > safeMaxChar
          || c < safeMinChar) {
        return escapeSlow(s, i);
      }
    }
    return s;
  }