public LineElement generateInnerElement(IBond bond, IRing ring, RendererModel model) {
        Point2d center = GeometryUtil.get2DCenter(ring);
        Point2d a = bond.getBegin().getPoint2d();
        Point2d b = bond.getEnd().getPoint2d();

        // the proportion to move in towards the ring center
        double distanceFactor = model.getParameter(TowardsRingCenterProportion.class).getValue();
        double ringDistance = distanceFactor * IDEAL_RINGSIZE / ring.getAtomCount();
        if (ringDistance < distanceFactor / MIN_RINGSIZE_FACTOR) ringDistance = distanceFactor / MIN_RINGSIZE_FACTOR;

        Point2d w = new Point2d();
        w.interpolate(a, center, ringDistance);
        Point2d u = new Point2d();
        u.interpolate(b, center, ringDistance);

        double alpha = 0.2;
        Point2d ww = new Point2d();
        ww.interpolate(w, u, alpha);
        Point2d uu = new Point2d();
        uu.interpolate(u, w, alpha);

        double width = getWidthForBond(bond, model);
        Color color = getColorForBond(bond, model);

        return new LineElement(u.x, u.y, w.x, w.y, width, color);
    }