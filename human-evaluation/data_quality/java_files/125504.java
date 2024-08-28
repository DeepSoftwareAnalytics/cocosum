private void writeListFile(final File inputfile, final String relativeRootFile) {
        try (Writer bufferedWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(inputfile)))) {
            bufferedWriter.write(relativeRootFile);
            bufferedWriter.flush();
        } catch (final IOException e) {
            logger.error(e.getMessage(), e) ;
        }
    }