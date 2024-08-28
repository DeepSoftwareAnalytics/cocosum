List<String> generateTypes(List<MessageType> msgTypes) {
        return msgTypes.stream().map(t -> t.accept(stringVisitor, null)).collect(Collectors.toList());
    }