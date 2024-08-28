public int getArgmaxConfigId() {
        int argmax = -1;
        double max = s.minValue();
        for (int c = 0; c < this.size(); c++) {
            double val = getValue(c);
            if (s.gte(val, max)) {
                max = val;
                argmax = c;
            }
        }
        return argmax;
    }