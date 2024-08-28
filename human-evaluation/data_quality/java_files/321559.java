@Override
    public final HeadDocument findByFileAndString(
            final String filename, final String string) {
        final Query searchQuery = new Query(Criteria.where("string").is(string)
                .and("filename").is(filename));
        final HeadDocument headDocument =
                mongoTemplate.findOne(searchQuery, HeadDocumentMongo.class);
        if (headDocument == null) {
            return null;
        }
        final Head head =
                (Head) toObjConverter.createGedObject(null, headDocument);
        headDocument.setGedObject(head);
        return headDocument;
    }