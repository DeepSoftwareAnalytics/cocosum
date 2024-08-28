public /*@Nullable*/Downloader startGetFile(final String path, /*@Nullable*/String rev)
            throws DbxException
    {
        DbxPathV1.checkArgNonRoot("path", path);
        String apiPath = "1/files/auto" + path;
        /*@Nullable*/String[] params = {
            "rev", rev
        };
        return startGetSomething(apiPath, params);
    }