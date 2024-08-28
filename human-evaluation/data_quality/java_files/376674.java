protected void setOwnerId(Integer id){
        if(getOwnerId() != null){
            logger.warn("Attempting to set owner id for an object where it as previously been set. Ignoring new id");
            return;
        }
        meta_put(OWNER_ID_KEY, id); // dont allow null
    }