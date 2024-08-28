private static void section1(List<PolymerNotation> polymers, Map<String, String> mapIds) throws NotationException {
    for (PolymerNotation polymer : polymers) {
      if (mapIds.containsKey(polymer.getPolymerID().getId())) {
        /* change id */
        PolymerNotation newpolymer = new PolymerNotation(mapIds.get(polymer.getPolymerID().getId()));
        newpolymer = new PolymerNotation(newpolymer.getPolymerID(), polymer.getPolymerElements());
        helm2notation.addPolymer(newpolymer);
      } else {
        helm2notation.addPolymer(polymer);
      }

    }

  }