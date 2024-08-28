public void sync() throws ClosedJournalException, IOException {
        try {
            appender.sync().get();
            if (appender.getAsyncException() != null) {
                throw new IOException(appender.getAsyncException());
            }
        } catch (Exception ex) {
            throw new IllegalStateException(ex.getMessage(), ex);
        }
    }