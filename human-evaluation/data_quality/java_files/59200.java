public void marshall(ApiKey apiKey, ProtocolMarshaller protocolMarshaller) {

        if (apiKey == null) {
            throw new SdkClientException("Invalid argument passed to marshall(...)");
        }

        try {
            protocolMarshaller.marshall(apiKey.getId(), ID_BINDING);
            protocolMarshaller.marshall(apiKey.getDescription(), DESCRIPTION_BINDING);
            protocolMarshaller.marshall(apiKey.getExpires(), EXPIRES_BINDING);
        } catch (Exception e) {
            throw new SdkClientException("Unable to marshall request to JSON: " + e.getMessage(), e);
        }
    }