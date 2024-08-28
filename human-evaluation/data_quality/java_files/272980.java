@NonNull
    public Expression collate(@NonNull Collation collation) {
        if (collation == null) {
            throw new IllegalArgumentException("collation cannot be null.");
        }
        return new CollationExpression(this, collation);
    }