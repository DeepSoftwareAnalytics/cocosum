public ResponseEntity handleWebFingerDiscoveryRequest(final String resource, final String rel) {
        if (StringUtils.isNotBlank(rel) && !OidcConstants.WEBFINGER_REL.equalsIgnoreCase(rel)) {
            LOGGER.warn("Handling discovery request for a non-standard OIDC relation [{}]", rel);
        }

        val issuer = this.discovery.getIssuer();
        if (!StringUtils.equalsIgnoreCase(resource, issuer)) {
            val resourceUri = normalize(resource);
            if (resourceUri == null) {
                LOGGER.error("Unable to parse and normalize resource: [{}]", resource);
                return buildNotFoundResponseEntity("Unable to normalize provided resource");
            }
            val issuerUri = normalize(issuer);
            if (issuerUri == null) {
                LOGGER.error("Unable to parse and normalize issuer: [{}]", issuer);
                return buildNotFoundResponseEntity("Unable to normalize issuer");
            }

            if (!"acct".equals(resourceUri.getScheme())) {
                LOGGER.error("Unable to accept resource scheme: [{}]", resourceUri.toUriString());
                return buildNotFoundResponseEntity("Unable to recognize/accept resource scheme " + resourceUri.getScheme());
            }

            var user = userInfoRepository.findByEmailAddress(resourceUri.getUserInfo() + '@' + resourceUri.getHost());
            if (user.isEmpty()) {
                user = userInfoRepository.findByUsername(resourceUri.getUserInfo());
            }
            if (user.isEmpty()) {
                LOGGER.info("User/Resource not found: [{}]", resource);
                return buildNotFoundResponseEntity("Unable to find resource");
            }

            if (!StringUtils.equalsIgnoreCase(issuerUri.getHost(), resourceUri.getHost())) {
                LOGGER.info("Host mismatch for resource [{}]: expected [{}] and yet received [{}]", resource,
                    issuerUri.getHost(), resourceUri.getHost());
                return buildNotFoundResponseEntity("Unable to match resource host");
            }
        }

        val body = new LinkedHashMap<String, Object>();
        body.put("subject", resource);

        val links = new ArrayList<>();
        links.add(CollectionUtils.wrap("rel", OidcConstants.WEBFINGER_REL, "href", issuer));
        body.put("links", links);

        return new ResponseEntity<>(body, HttpStatus.OK);
    }