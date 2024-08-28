static public Key[] exit(Key... keep) {
    List<Key> keylist = new ArrayList<>();
    if( keep != null )
      for( Key k : keep ) if (k != null) keylist.add(k);
    Object[] arrkeep = keylist.toArray();
    Arrays.sort(arrkeep);
    Stack<HashSet<Key>> keys = _scope.get()._keys;
    if (keys.size() > 0) {
      Futures fs = new Futures();
      for (Key key : keys.pop()) {
        int found = Arrays.binarySearch(arrkeep, key);
        if ((arrkeep.length == 0 || found < 0) && key != null) Keyed.remove(key, fs);
      }
      fs.blockForPending();
    }
    return keep;
  }