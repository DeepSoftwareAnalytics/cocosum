public void resetLayerDefaultConfig() {
        //clear the learning related params for all layers in the origConf and set to defaults
        this.setIUpdater(null);
        this.setWeightInitFn(null);
        this.setBiasInit(Double.NaN);
        this.setGainInit(Double.NaN);
        this.regularization = null;
        this.regularizationBias = null;
        this.setGradientNormalization(GradientNormalization.None);
        this.setGradientNormalizationThreshold(1.0);
        this.iUpdater = null;
        this.biasUpdater = null;
    }