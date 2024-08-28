protected VocabCache<ShallowSequenceElement> buildShallowVocabCache(Counter<Long> counter) {

        // TODO: need simplified cache here, that will operate on Long instead of string labels
        VocabCache<ShallowSequenceElement> vocabCache = new AbstractCache<>();
        for (Long id : counter.keySet()) {
            ShallowSequenceElement shallowElement = new ShallowSequenceElement(counter.getCount(id), id);
            vocabCache.addToken(shallowElement);
        }

        // building huffman tree
        Huffman huffman = new Huffman(vocabCache.vocabWords());
        huffman.build();
        huffman.applyIndexes(vocabCache);

        return vocabCache;
    }