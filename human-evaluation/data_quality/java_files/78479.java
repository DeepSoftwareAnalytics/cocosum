public static Collection<Set<ProposalResponse>> getProposalConsistencySets(Collection<? extends ProposalResponse> proposalResponses,
                                                                               Set<ProposalResponse> invalid) throws InvalidArgumentException {

        if (proposalResponses == null) {
            throw new InvalidArgumentException("proposalResponses collection is null");
        }

        if (proposalResponses.isEmpty()) {
            throw new InvalidArgumentException("proposalResponses collection is empty");
        }

        if (null == invalid) {
            throw new InvalidArgumentException("invalid set is null.");
        }

        HashMap<ByteString, Set<ProposalResponse>> ret = new HashMap<>();

        for (ProposalResponse proposalResponse : proposalResponses) {

            if (proposalResponse.isInvalid()) {
                invalid.add(proposalResponse);
            } else {
                // payload bytes is what's being signed over so it must be consistent.
                final ByteString payloadBytes = proposalResponse.getPayloadBytes();

                if (payloadBytes == null) {
                    throw new InvalidArgumentException(format("proposalResponse.getPayloadBytes() was null from peer: %s.",
                            proposalResponse.getPeer()));
                } else if (payloadBytes.isEmpty()) {
                    throw new InvalidArgumentException(format("proposalResponse.getPayloadBytes() was empty from peer: %s.",
                            proposalResponse.getPeer()));
                }
                Set<ProposalResponse> set = ret.computeIfAbsent(payloadBytes, k -> new HashSet<>());
                set.add(proposalResponse);
            }
        }

        if (IS_DEBUG_LEVEL && ret.size() > 1) {

            StringBuilder sb = new StringBuilder(1000);

            int i = 0;
            String sep = "";

            for (Map.Entry<ByteString, Set<ProposalResponse>> entry : ret.entrySet()) {
                ByteString bytes = entry.getKey();
                Set<ProposalResponse> presp = entry.getValue();

                sb.append(sep)
                        .append("Consistency set: ").append(i++).append(" bytes size: ").append(bytes.size())
                        .append(" bytes: ")
                        .append(Utils.toHexString(bytes.toByteArray())).append(" [");

                String psep = "";

                for (ProposalResponse proposalResponse : presp) {
                    sb.append(psep).append(proposalResponse.getPeer());
                    psep = ", ";
                }
                sb.append("]");
                sep = ", ";
            }

            logger.debug(sb.toString());

        }

        return ret.values();

    }