private boolean parseRemoteDomainControllerAttributes_1_5(final XMLExtendedStreamReader reader, final ModelNode address,
                                                              final List<ModelNode> list, boolean allowDiscoveryOptions) throws XMLStreamException {

        final ModelNode update = new ModelNode();
        update.get(OP_ADDR).set(address);
        update.get(OP).set(RemoteDomainControllerAddHandler.OPERATION_NAME);

        // Handle attributes
        AdminOnlyDomainConfigPolicy adminOnlyPolicy = AdminOnlyDomainConfigPolicy.DEFAULT;
        boolean requireDiscoveryOptions = false;
        final int count = reader.getAttributeCount();
        for (int i = 0; i < count; i++) {
            final String value = reader.getAttributeValue(i);
            if (!isNoNamespaceAttribute(reader, i)) {
                throw unexpectedAttribute(reader, i);
            } else {
                final Attribute attribute = Attribute.forName(reader.getAttributeLocalName(i));
                switch (attribute) {
                    case HOST: {
                        DomainControllerWriteAttributeHandler.HOST.parseAndSetParameter(value, update, reader);
                        break;
                    }
                    case PORT: {
                        DomainControllerWriteAttributeHandler.PORT.parseAndSetParameter(value, update, reader);
                        break;
                    }
                    case SECURITY_REALM: {
                        DomainControllerWriteAttributeHandler.SECURITY_REALM.parseAndSetParameter(value, update, reader);
                        break;
                    }
                    case USERNAME: {
                        DomainControllerWriteAttributeHandler.USERNAME.parseAndSetParameter(value, update, reader);
                        break;
                    }
                    case ADMIN_ONLY_POLICY: {
                        DomainControllerWriteAttributeHandler.ADMIN_ONLY_POLICY.parseAndSetParameter(value, update, reader);
                        ModelNode nodeValue = update.get(DomainControllerWriteAttributeHandler.ADMIN_ONLY_POLICY.getName());
                        if (nodeValue.getType() != ModelType.EXPRESSION) {
                            adminOnlyPolicy = AdminOnlyDomainConfigPolicy.getPolicy(nodeValue.asString());
                        }
                        break;
                    }
                    default:
                        throw unexpectedAttribute(reader, i);
                }
            }
        }

        if (!update.hasDefined(DomainControllerWriteAttributeHandler.HOST.getName())) {
            if (allowDiscoveryOptions) {
                requireDiscoveryOptions = isRequireDiscoveryOptions(adminOnlyPolicy);
            } else {
                throw ParseUtils.missingRequired(reader, Collections.singleton(Attribute.HOST.getLocalName()));
            }
        }
        if (!update.hasDefined(DomainControllerWriteAttributeHandler.PORT.getName())) {
            if (allowDiscoveryOptions) {
                requireDiscoveryOptions = requireDiscoveryOptions || isRequireDiscoveryOptions(adminOnlyPolicy);
            } else {
                throw ParseUtils.missingRequired(reader, Collections.singleton(Attribute.PORT.getLocalName()));
            }
        }

        list.add(update);
        return requireDiscoveryOptions;
    }