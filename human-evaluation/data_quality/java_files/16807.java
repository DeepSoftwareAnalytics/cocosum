public Map<String, INDArray> feedForward(INDArray input, boolean train) {
        if (numInputArrays != 1)
            throw new UnsupportedOperationException("Cannot feedForward with single input for graph network with "
                    + numInputArrays + " expected inputs");
        setInput(0, input);
        return feedForward(train);
    }