public FieldFilterBuilder isDefault() {
        add(new NegationFieldFilter(new ModifierFieldFilter(Modifier.PUBLIC & Modifier.PROTECTED & Modifier.PRIVATE)));
        return this;
    }