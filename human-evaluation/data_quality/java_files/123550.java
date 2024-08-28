public EasyRandomParameters stringLengthRange(final int minStringLength, final int maxStringLength) {
        if (minStringLength < 0) {
            throw new IllegalArgumentException("minStringLength must be >= 0");
        }
        if (minStringLength > maxStringLength) {
            throw new IllegalArgumentException(format("minStringLength (%s) must be <= than maxStringLength (%s)",
                    minStringLength, maxStringLength));
        }
        setStringLengthRange(new Range<>(minStringLength, maxStringLength));
        return this;
    }