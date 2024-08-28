@Override
    public Object getConnection( Subject subject,
                                 ConnectionRequestInfo cxRequestInfo ) throws ResourceException {
        JcrSessionHandle handle = new JcrSessionHandle(this);
        addHandle(handle);
        return handle;
    }