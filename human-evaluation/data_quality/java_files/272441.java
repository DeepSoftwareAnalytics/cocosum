CssFormatter endBlock() {
        blockDeep--;
        if( blockDeep == 0 ) {
            state.pool.free( insets );
            insets = null;
            inlineMode = false;
        } else {
            if( blockDeep == 1 && currentOutput.getClass() == CssMediaOutput.class ) {
                state.pool.free( insets );
                insets = null;
                inlineMode = false;
            } else {
                endBlockImpl();
            }
        }
        return this;
    }