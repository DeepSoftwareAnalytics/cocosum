public static String extractPackageName(TypeElement element) {
		String fullName = element.getQualifiedName().toString();

		if (fullName.lastIndexOf(".") > 0) {
			return fullName.substring(0, fullName.lastIndexOf("."));
		}
		return "";
	}