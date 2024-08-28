@Override
	public Iterator<E> iterator() {
		combine();
		if(combined==null) {
			Set<E> emptySet = java.util.Collections.emptySet();
			return emptySet.iterator();
		}
		return combined.iterator();
	}