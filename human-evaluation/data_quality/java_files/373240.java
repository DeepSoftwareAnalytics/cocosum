public static final String removePathPrefix(File prefix, File file) {
		final String r = file.getAbsolutePath().replaceFirst(
				"^" //$NON-NLS-1$
				+ Pattern.quote(prefix.getAbsolutePath()),
				EMPTY_STRING);
		if (r.startsWith(File.separator)) {
			return r.substring(File.separator.length());
		}
		return r;
	}