public KeyValueStoreUpdate clearAll(String... segments) {
		actions.add(new Deletion("/" + String.join("/", segments)));
		return this;
	}