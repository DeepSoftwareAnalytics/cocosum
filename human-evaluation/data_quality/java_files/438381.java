public static Class isAssignableTo(final Class<?> reference, final Class<?> toValue, final String message) {
        return precondition(reference, new Predicate<Class>() {
            @Override
            public boolean test(Class testValue) {
                return toValue.isAssignableFrom(testValue);
            }
        }, ClassCastException.class, message);
    }