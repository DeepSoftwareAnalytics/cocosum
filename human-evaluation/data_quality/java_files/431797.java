public static <T, K, S, X extends Collection<T>> X loadNToOne(X list,
			Function<T, K> keyGetter,
			Function<Collection<K>, Collection<S>> loader,
			Function<S, K> loadedKeyGetter, BiConsumer<T, S> match) {
		if (list == null || list.size() == 0)
			return list;

		Set<K> keys = toSet(list, keyGetter);
		if (keys == null || keys.size() == 0)
			return list;

		Collection<S> subs = loader.apply(keys);
		if (subs == null || subs.size() == 0)
			return list;

		HashMap<K, S> map = toHashMap(subs, loadedKeyGetter);
		for (T item : list) {
			K key = keyGetter.apply(item);
			if (key == null)
				continue;

			S sub = map.get(key);
			if (sub == null)
				continue;
			match.accept(item, sub);
		}

		return list;
	}