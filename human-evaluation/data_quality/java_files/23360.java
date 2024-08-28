public static Sampler create(Double selectPercent) {
		if (selectPercent.equals(ALWAYS)) {
			return new AlwaysSampler();
		} else if (selectPercent.equals(NEVER)) {
			return new NeverSampler();
		} else {
			return new Sampler(selectPercent);
		}
	}