public int nullity() {
        if (!full) {
            throw new IllegalStateException("This is not a FULL singular value decomposition.");
        }

        int r = 0;
        for (int i = 0; i < s.length; i++) {
            if (s[i] <= tol) {
                r++;
            }
        }
        return r;
    }