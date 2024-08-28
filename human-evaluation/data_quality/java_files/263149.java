private Map<String,SemanticVector> loadText(File sspaceFile) 
        throws IOException {

        LOGGER.info("loading text TSS from " + sspaceFile);
        
        BufferedReader br = new BufferedReader(new FileReader(sspaceFile));
        String[] header = br.readLine().split("\\s+");
        int words = Integer.parseInt(header[0]);
        dimensions = Integer.parseInt(header[1]);
        
        Map<String,SemanticVector> wordToSemantics = 
            new HashMap<String,SemanticVector>(words, 2f);
        
        // read in each word
        for (String line = null; (line = br.readLine()) != null; ) {
            String[] wordAndSemantics = line.split("\\|");
            String word = wordAndSemantics[0];
            SemanticVector semantics = new SemanticVector(dimensions);

            LOGGER.info("loading " + wordAndSemantics.length + 
                " timesteps for word " + word); 

            for (int i = 1; i < wordAndSemantics.length; ++i) {
                String[] timeStepAndValues = wordAndSemantics[i].split(" ");
                long timeStep = Long.parseLong(timeStepAndValues[0]);
                updateTimeRange(timeStep);

                // Load that time step's vector.  Note that we make the
                // assumption here that even though the T-Space is serialized in
                // a dense format, that the vector data is actually sparse, and
                // so it will be more efficient to store it as such.
                Map<Integer,Double> sparseArray = new IntegerMap<Double>();

                for (int j = 1; j < timeStepAndValues.length; ++j) {
                    sparseArray.put(Integer.valueOf(j-1), 
                            Double.valueOf(timeStepAndValues[j]));
                }
                semantics.setSemantics(timeStep, sparseArray);
            }
            wordToSemantics.put(word,semantics);
        }
    
        return wordToSemantics;
    }