protected static void validateBool(String opName, SDVariable v) {
        if (v == null)
            return;
        if (v.dataType() != DataType.BOOL)
            throw new IllegalStateException("Cannot apply operation \"" + opName + "\" to variable \"" + v.getVarName() + "\" with non-boolean point data type " + v.dataType());
    }