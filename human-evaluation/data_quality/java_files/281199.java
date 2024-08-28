protected boolean ensureInput(int minAmount)
        throws XMLStreamException
    {
        int currAmount = mInputEnd - mInputPtr;
        if (currAmount >= minAmount) {
            return true;
        }
        try {
            return mInput.readMore(this, minAmount);
        } catch (IOException ie) {
            throw constructFromIOE(ie);
        }
    }