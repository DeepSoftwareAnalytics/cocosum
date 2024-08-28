@Override
    public ApplicationDescriptor connectorModule(String uri) {
        model.createChild("module").createChild("connector").text(uri);
        return this;
    }