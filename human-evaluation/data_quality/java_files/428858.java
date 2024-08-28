protected void doDelete(String spaceId, String key) throws IOException {
        Term termDelete = buildIdTerm(spaceId, key);
        IndexWriter indexWriter = getIndexWriter();
        indexWriter.deleteDocuments(termDelete);
        if (!isAsyncWrite()) {
            indexWriter.commit();
        } else if (getIndexManager().getBackgroundCommitIndexPeriodMs() <= 0) {
            LOGGER.warn("Async-write is enable, autoCommitPeriodMs must be larger than 0!");
        }
    }