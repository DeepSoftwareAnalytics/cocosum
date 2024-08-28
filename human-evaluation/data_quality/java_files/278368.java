public CollectionUpdateResponse updateCollection(CollectionUpdateRequest request) throws IOException {
        return new CollectionUpdateResponse(client.doPost(request.toUrl(), request.getBody()));
    }