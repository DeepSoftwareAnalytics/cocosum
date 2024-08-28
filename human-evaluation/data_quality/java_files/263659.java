public void setOrderFlags(final int flags) {
		orderFlags = Arrays.
				stream(BitfinexOrderFlag.values())
				.filter(f -> ((f.getFlag() & flags) == f.getFlag()))
				.collect(Collectors.toSet());
	}