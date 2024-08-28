public void complete() {

        m_completed = true;
        // Prevent changing of the cached lists
        if (m_headers != null) {
            m_headers = Collections.unmodifiableMap(m_headers);
        }
        if (m_elements != null) {
            m_elements = Collections.unmodifiableList(m_elements);
        }
        if (LOG.isDebugEnabled()) {
            LOG.debug(Messages.get().getBundle().key(Messages.LOG_FLEXCACHEENTRY_ENTRY_COMPLETED_1, toString()));
        }
    }