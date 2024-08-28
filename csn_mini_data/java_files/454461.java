public static CharSource wrap(CharSequence charSequence) {
    return charSequence instanceof String
        ? new StringCharSource((String) charSequence)
        : new CharSequenceCharSource(charSequence);
  }