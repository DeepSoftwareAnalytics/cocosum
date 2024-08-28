public static File getCanonicalFile(final File file) {

        // Check sanity
        Validate.notNull(file, "file");

        // All done
        try {
            return file.getCanonicalFile();
        } catch (IOException e) {
            throw new IllegalArgumentException("Could not acquire the canonical file for ["
                    + file.getAbsolutePath() + "]", e);
        }
    }