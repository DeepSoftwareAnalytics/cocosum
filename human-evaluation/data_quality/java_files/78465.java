private JsonObject toJsonObject() {
        JsonObjectBuilder factory = Json.createObjectBuilder();
        if (idemixCredReq != null) {
            factory.add("request", idemixCredReq.toJsonObject());
        } else {
            factory.add("request", JsonValue.NULL);
        }
        if (caName != null) {
            factory.add(HFCAClient.FABRIC_CA_REQPROP, caName);
        }
        return factory.build();
    }