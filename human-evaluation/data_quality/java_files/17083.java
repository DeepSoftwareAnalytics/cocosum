public static INDArray im2col(INDArray img, int kh, int kw, int sy, int sx, int ph, int pw, int pval,
                                  boolean isSameMode) {
        INDArray output = null;

        if (isSameMode) {
            int oH = (int) Math.ceil(img.size(2) * 1.f / sy);
            int oW = (int) Math.ceil(img.size(3) * 1.f / sx);

            output = Nd4j.createUninitialized(img.dataType(), new long[]{img.size(0), img.size(1), kh, kw, oH, oW}, 'c');
        } else {
            // FIXME: int cast
            int oH = ((int) img.size(2) - (kh + (kh - 1) * (1 - 1)) + 2 * ph) / sy + 1;
            int oW = ((int) img.size(3) - (kw + (kw - 1) * (1 - 1)) + 2 * pw) / sx + 1;

            output = Nd4j.createUninitialized(img.dataType(), new long[]{img.size(0), img.size(1), kh, kw, oH, oW}, 'c');
        }

        Im2col im2col = Im2col.builder()
                .inputArrays(new INDArray[]{img})
                .outputs(new INDArray[]{output})
                .conv2DConfig(Conv2DConfig.builder()
                        .pW(pw)
                        .pH(ph)
                        .sH(sy)
                        .sW(sx)
                        .kW(kw)
                        .kH(kh)
                        .dW(1)
                        .dH(1)
                        .isSameMode(isSameMode)
                        .build()).build();

        Nd4j.getExecutioner().execAndReturn(im2col);
        return im2col.outputArguments()[0];
    }