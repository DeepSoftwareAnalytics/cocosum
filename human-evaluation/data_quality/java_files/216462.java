protected void deleteResource(I_CmsIndexWriter indexWriter, CmsPublishedResource resource) {

        try {
            if (LOG.isInfoEnabled()) {
                LOG.info(Messages.get().getBundle().key(Messages.LOG_DELETING_FROM_INDEX_1, resource.getRootPath()));
            }
            // delete all documents with this term from the index
            indexWriter.deleteDocument(resource);
        } catch (IOException e) {
            if (LOG.isWarnEnabled()) {
                LOG.warn(
                    Messages.get().getBundle().key(
                        Messages.LOG_IO_INDEX_DOCUMENT_DELETE_2,
                        resource.getRootPath(),
                        m_index.getName()),
                    e);
            }
        }
    }