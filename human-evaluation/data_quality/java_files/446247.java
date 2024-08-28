public static String getExtension(String fileName) {
        int dotIndex = fileName.lastIndexOf(DOT);
        return dotIndex > 0 ? fileName.substring(dotIndex) : EMPTY;
    }