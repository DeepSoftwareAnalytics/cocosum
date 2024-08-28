public OrmExtracted fromDomain(T domainObject, Namespace namespace, BinaryValue key)
    {
        BinaryValue bucketName = namespace != null ? namespace.getBucketName() : null;
        BinaryValue bucketType = namespace != null ? namespace.getBucketType() : null;

        key = AnnotationUtil.getKey(domainObject, key);
        bucketName = AnnotationUtil.getBucketName(domainObject, bucketName);
        bucketType = AnnotationUtil.getBucketType(domainObject, bucketType);

        if (bucketName == null)
        {
            throw new ConversionException("Bucket name not provided via namespace or domain object");
        }

        VClock vclock = AnnotationUtil.getVClock(domainObject);
        String contentType =
            AnnotationUtil.getContentType(domainObject, RiakObject.DEFAULT_CONTENT_TYPE);

        RiakObject riakObject = new RiakObject();

        AnnotationUtil.getUsermetaData(riakObject.getUserMeta(), domainObject);
        AnnotationUtil.getIndexes(riakObject.getIndexes(), domainObject);
        AnnotationUtil.getLinks(riakObject.getLinks(), domainObject);

        ContentAndType cAndT = fromDomain(domainObject);
        contentType = cAndT.contentType != null ? cAndT.contentType : contentType;

        riakObject.setContentType(contentType)
                    .setValue(cAndT.content)
                    .setVClock(vclock);

        // We allow an annotated object to omit @BucketType and get the default
        Namespace ns;
        if (bucketType == null)
        {
            ns = new Namespace(bucketName);
        }
        else
        {
            ns = new Namespace(bucketType, bucketName);
        }

        OrmExtracted extracted = new OrmExtracted(riakObject, ns, key);
        return extracted;
    }