public void marshall(Bucket bucket, ProtocolMarshaller protocolMarshaller) {

        if (bucket == null) {
            throw new SdkClientException("Invalid argument passed to marshall(...)");
        }

        try {
            protocolMarshaller.marshall(bucket.getValue(), VALUE_BINDING);
            protocolMarshaller.marshall(bucket.getCount(), COUNT_BINDING);
        } catch (Exception e) {
            throw new SdkClientException("Unable to marshall request to JSON: " + e.getMessage(), e);
        }
    }