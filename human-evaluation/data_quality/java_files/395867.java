public static void decodeCorrelationIdentifiers(Set<CorrelationIdentifier> correlations, String encoded) {
        if (encoded != null && !encoded.trim().isEmpty()) {
            StringTokenizer st = new StringTokenizer(encoded, ",");
            while (st.hasMoreTokens()) {
                String token = st.nextToken();
                String[] parts = token.split("[|]");
                if (parts.length == 2) {
                    String scope = parts[0].trim();
                    String value = parts[1].trim();

                    log.tracef("Extracted correlation identifier scope [%s] value [%s]", scope, value);

                    CorrelationIdentifier cid = new CorrelationIdentifier();
                    cid.setScope(Scope.valueOf(scope));
                    cid.setValue(value);

                    correlations.add(cid);
                }
            }
        }
    }