@Override
	public T convert(String name) throws IllegalArgumentException {
		if (name == null) {
			throw new IllegalArgumentException("Cant convert 'null' to " + enumClass.getSimpleName());
		}
		try {
			return Enum.valueOf(enumClass, name);
		} catch (IllegalArgumentException e) {
			// ignore
		}
		try {
			return Enum.valueOf(enumClass, name.toUpperCase());
		} catch (IllegalArgumentException e) {
			// ignore
		}
		try {
			return Enum.valueOf(enumClass, name.toUpperCase().replace("-", "_"));
		} catch (IllegalArgumentException e) {
			// ignore
		}
		throw new IllegalArgumentException("Can't convert " + name + " to " + enumClass.getSimpleName());

	}