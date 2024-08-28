public INDArray computeGaussianPerplexity(final INDArray d, double u) {
        N = d.rows();

        final int k = (int) (3 * u);
        if (u > k)
            throw new IllegalStateException("Illegal k value " + k + "greater than " + u);


        rows = zeros(1, N + 1);
        cols = zeros(1, N * k);
        vals = zeros(1, N * k);

        for (int n = 0; n < N; n++)
            rows.putScalar(n + 1, rows.getDouble(n) + k);


        final INDArray beta = ones(N, 1);

        final double logU = FastMath.log(u);
        VPTree tree = new VPTree(d, simiarlityFunction, vpTreeWorkers,invert);

        MemoryWorkspace workspace =
                workspaceMode == WorkspaceMode.NONE ? new DummyWorkspace()
                        : Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(
                        workspaceConfigurationExternal,
                        workspaceExternal);
        try (MemoryWorkspace ws = workspace.notifyScopeEntered()) {
            log.info("Calculating probabilities of data similarities...");
            for (int i = 0; i < N; i++) {
                if (i % 500 == 0)
                    log.info("Handled " + i + " records");

                double betaMin = -Double.MAX_VALUE;
                double betaMax = Double.MAX_VALUE;
                List<DataPoint> results = new ArrayList<>();
                tree.search(d.slice(i), k + 1, results, new ArrayList<Double>());
                double betas = beta.getDouble(i);

                if(results.size() == 0){
                    throw new IllegalStateException("Search returned no values for vector " + i +
                            " - similarity \"" + simiarlityFunction + "\" may not be defined (for example, vector is" +
                            " all zeros with cosine similarity)");
                }

                INDArray cArr = VPTree.buildFromData(results);
                Pair<INDArray, Double> pair = computeGaussianKernel(cArr, beta.getDouble(i), k);
                INDArray currP = pair.getFirst();
                double hDiff = pair.getSecond() - logU;
                int tries = 0;
                boolean found = false;
                //binary search
                while (!found && tries < 200) {
                    if (hDiff < tolerance && -hDiff < tolerance)
                        found = true;
                    else {
                        if (hDiff > 0) {
                            betaMin = betas;

                            if (betaMax == Double.MAX_VALUE || betaMax == -Double.MAX_VALUE)
                                betas *= 2;
                            else
                                betas = (betas + betaMax) / 2.0;
                        } else {
                            betaMax = betas;
                            if (betaMin == -Double.MAX_VALUE || betaMin == Double.MAX_VALUE)
                                betas /= 2.0;
                            else
                                betas = (betas + betaMin) / 2.0;
                        }

                        pair = computeGaussianKernel(cArr, betas, k);
                        hDiff = pair.getSecond() - logU;
                        tries++;
                    }

                }


                currP.divi(currP.sum(Integer.MAX_VALUE));
                INDArray indices = Nd4j.create(1, k + 1);
                for (int j = 0; j < indices.length(); j++) {
                    if (j >= results.size())
                        break;
                    indices.putScalar(j, results.get(j).getIndex());
                }

                for (int l = 0; l < k; l++) {
                    cols.putScalar(rows.getInt(i) + l, indices.getDouble(l + 1));
                    vals.putScalar(rows.getInt(i) + l, currP.getDouble(l));
                }
            }
        }
        return vals;

    }