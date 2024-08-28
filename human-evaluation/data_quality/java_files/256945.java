public AndCondition copy() {
        AndCondition builder = new AndCondition();
        builder.children.addAll(children);
        return builder;
    }