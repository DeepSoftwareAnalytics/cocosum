public static List<String> getLayersDigests(String manifestContent) throws IOException {
        List<String> dockerLayersDependencies = new ArrayList<String>();

        JsonNode manifest = Utils.mapper().readTree(manifestContent);
        JsonNode schemaVersion = manifest.get("schemaVersion");
        if (schemaVersion == null) {
            throw new IllegalStateException("Could not find 'schemaVersion' in manifest");
        }

        boolean isSchemeVersion1 = schemaVersion.asInt() == 1;
        JsonNode fsLayers = getFsLayers(manifest, isSchemeVersion1);
        for (JsonNode fsLayer : fsLayers) {
            JsonNode blobSum = getBlobSum(isSchemeVersion1, fsLayer);
            dockerLayersDependencies.add(blobSum.asText());
        }
        dockerLayersDependencies.add(getConfigDigest(manifestContent));

        //Add manifest sha1
        String manifestSha1 = Hashing.sha1().hashString(manifestContent, Charsets.UTF_8).toString();
        dockerLayersDependencies.add("sha1:" + manifestSha1);

        return dockerLayersDependencies;
    }