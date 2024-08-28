@VisibleForTesting
  ValueReference<K, V> newValueReference(ReferenceEntry<K, V> entry, V value) {
    int hash = entry.getHash();
    return valueStrength.referenceValue(segmentFor(hash), entry, value);
  }