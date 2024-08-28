@Override
  protected final int nextEscapeIndex(CharSequence csq, int index, int end) {
    while (index < end) {
      char c = csq.charAt(index);
      if ((c < replacementsLength && replacements[c] != null)
          || c > safeMaxChar
          || c < safeMinChar) {
        break;
      }
      index++;
    }
    return index;
  }