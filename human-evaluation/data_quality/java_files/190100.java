public void changeBase(Permutation newBase) {
        PermutationGroup h = new PermutationGroup(newBase);

        int firstDiffIndex = base.firstIndexOfDifference(newBase);

        for (int j = firstDiffIndex; j < size; j++) {
            for (int a = 0; a < size; a++) {
                Permutation g = permutations[j][a];
                if (g != null) {
                    h.enter(g);
                }
            }
        }

        for (int j = 0; j < firstDiffIndex; j++) {
            for (int a = 0; a < size; a++) {
                Permutation g = permutations[j][a];
                if (g != null) {
                    int hj = h.base.get(j);
                    int x = g.get(hj);
                    h.permutations[j][x] = new Permutation(g);
                }
            }
        }
        this.base = new Permutation(h.base);
        this.permutations = h.permutations.clone();
    }