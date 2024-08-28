public static WindowOver<Double> regrSxy(Expression<? extends Number> arg1, Expression<? extends Number> arg2) {
        return new WindowOver<Double>(Double.class, SQLOps.REGR_SXY, arg1, arg2);
    }