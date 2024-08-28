@Override
  protected final char[] escape(int cp) {
    if (cp < replacementsLength) {
      char[] chars = replacements[cp];
      if (chars != null) {
        return chars;
      }
    }
    if (cp >= safeMin && cp <= safeMax) {
      return null;
    }
    return escapeUnsafe(cp);
  }