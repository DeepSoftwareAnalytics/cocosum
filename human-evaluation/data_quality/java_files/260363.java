public static String renderClassLine(final Class<?> type, final List<String> markers) {
        return String.format("%-28s %-26s %s", getClassName(type), brackets(renderPackage(type)), markers(markers));
    }