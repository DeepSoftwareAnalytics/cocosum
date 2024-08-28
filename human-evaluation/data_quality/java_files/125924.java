@Override
    public Short getShort(int index) throws IndexOutOfBoundsException {
        indexCheck();
        return (short) rows.get(row).getInteger(index).intValue();
    }