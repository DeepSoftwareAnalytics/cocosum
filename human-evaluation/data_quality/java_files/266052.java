private void checkLayout()
    {
        Dimension dim = getLayoutDimension();
        if (currentDimension == null || !currentDimension.equals(dim)) //the dimension has changed
        {
            createLayout(dim);
            //containing box for the new viewport
            viewport.setContainingBlockBox(owner);
            if (owner instanceof BlockBox)
                viewport.clipByBlock((BlockBox) owner);
            
            owner.removeAllSubBoxes();
            owner.addSubBox(viewport);
            currentDimension = new Dimension(dim);
        }
    }