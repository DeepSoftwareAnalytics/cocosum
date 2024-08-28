@SuppressWarnings("unchecked")
    protected final <ViewType extends View> ViewType findViewById(@IdRes final int viewId) {
        Condition.INSTANCE.ensureNotNull(currentParentView, "No parent view set",
                IllegalStateException.class);
        ViewHolder viewHolder = (ViewHolder) currentParentView.getTag();

        if (viewHolder == null) {
            viewHolder = new ViewHolder(currentParentView);
            currentParentView.setTag(viewHolder);
        }

        return (ViewType) viewHolder.findViewById(viewId);
    }