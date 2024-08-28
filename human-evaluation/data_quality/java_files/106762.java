public final boolean isCryptoAllowed(Key key)
            throws ExemptionMechanismException {
        boolean ret = false;
        if (done && (key != null)) {
            // Check if the key passed in is the same as the one
            // this exemption mechanism used.
            ret = keyStored.equals(key);
        }
        return ret;
     }