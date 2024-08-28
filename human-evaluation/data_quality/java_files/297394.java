@Override
    public String moveAsync(String apiKey, String from, String to) {
        checkNotNull(from, "from");
        checkNotNull(to, "to");
        try {
            URI uri = _databus.clone()
                    .segment("_move")
                    .queryParam("from", from)
                    .queryParam("to", to)
                    .build();
            Map<String, Object> response = _client.resource(uri)
                    .header(ApiKeyRequest.AUTHENTICATION_HEADER, apiKey)
                    .post(new TypeReference<Map<String, Object>>(){}, null);
            return response.get("id").toString();
        } catch (EmoClientException e) {
            throw convertException(e);
        }
    }