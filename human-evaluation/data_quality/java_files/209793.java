public static Object getValue(String registryName, String key) {
		Map<String, Object> registry = getRegistry(registryName);
		if (registry == null || StringUtils.isBlank(key)) {
			return null;
		}
		return registry.get(key);
	}