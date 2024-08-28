public void marshall(Condition condition, ProtocolMarshaller protocolMarshaller) {

        if (condition == null) {
            throw new SdkClientException("Invalid argument passed to marshall(...)");
        }

        try {
            protocolMarshaller.marshall(condition.getAttributeValueList(), ATTRIBUTEVALUELIST_BINDING);
            protocolMarshaller.marshall(condition.getComparisonOperator(), COMPARISONOPERATOR_BINDING);
        } catch (Exception e) {
            throw new SdkClientException("Unable to marshall request to JSON: " + e.getMessage(), e);
        }
    }