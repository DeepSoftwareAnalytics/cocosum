@UiThread
    private int getFlatParentPosition(int parentPosition) {
        int parentCount = 0;
        int listItemCount = mFlatItemList.size();
        for (int i = 0; i < listItemCount; i++) {
            if (mFlatItemList.get(i).isParent()) {
                parentCount++;

                if (parentCount > parentPosition) {
                    return i;
                }
            }
        }

        return INVALID_FLAT_POSITION;
    }