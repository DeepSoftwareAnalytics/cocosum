void expectIndexMatch(Node n, JSType objType, JSType indexType) {
    checkState(n.isGetElem() || n.isComputedProp(), n);
    Node indexNode = n.isGetElem() ? n.getLastChild() : n.getFirstChild();
    if (indexType.isSymbolValueType()) {
      // For now, allow symbols definitions/access on any type. In the future only allow them
      // on the subtypes for which they are defined.
      return;
    }
    if (objType.isStruct()) {
      report(JSError.make(indexNode,
                          ILLEGAL_PROPERTY_ACCESS, "'[]'", "struct"));
    }
    if (objType.isUnknownType()) {
      expectStringOrNumberOrSymbol(indexNode, indexType, "property access");
    } else {
      ObjectType dereferenced = objType.dereference();
      if (dereferenced != null && dereferenced
          .getTemplateTypeMap()
          .hasTemplateKey(typeRegistry.getObjectIndexKey())) {
        expectCanAssignTo(
            indexNode,
            indexType,
            dereferenced
                .getTemplateTypeMap()
                .getResolvedTemplateType(typeRegistry.getObjectIndexKey()),
            "restricted index type");
      } else if (dereferenced != null && dereferenced.isArrayType()) {
        expectNumberOrSymbol(indexNode, indexType, "array access");
      } else if (objType.matchesObjectContext()) {
        expectStringOrSymbol(indexNode, indexType, "property access");
      } else {
        mismatch(
            n,
            "only arrays or objects can be accessed",
            objType,
            typeRegistry.createUnionType(ARRAY_TYPE, OBJECT_TYPE));
      }
    }
  }