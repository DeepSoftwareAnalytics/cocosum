private List<Map.Entry<StreamName, BufferedStream>> cleanUpStreams() throws IOException {
    List<Map.Entry<StreamName, BufferedStream>> streamList =
      new ArrayList<Map.Entry<StreamName, BufferedStream>>(streams.size());
    Map<StreamName, Integer> indexMap = new HashMap<StreamName, Integer>(streams.size());

    int increment = 0;

    for(Map.Entry<StreamName, BufferedStream> pair: streams.entrySet()) {
      if (!pair.getValue().isSuppressed()) {
        StreamName name = pair.getKey();
        if (name.getKind() == Kind.LENGTH) {
          Integer index = indexMap.get(new StreamName(name.getColumn(), Kind.DICTIONARY_DATA));
          if (index != null) {
            streamList.add(index + increment, pair);
            increment++;
            continue;
          }
        }

        indexMap.put(name, new Integer(streamList.size()));
        streamList.add(pair);
      } else {
        pair.getValue().clear();
      }
    }

    return streamList;
  }