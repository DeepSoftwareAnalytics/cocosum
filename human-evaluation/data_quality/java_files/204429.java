public static PayloadReader reader(final String pathAccountSid, 
                                       final String pathReferenceSid, 
                                       final String pathAddOnResultSid) {
        return new PayloadReader(pathAccountSid, pathReferenceSid, pathAddOnResultSid);
    }