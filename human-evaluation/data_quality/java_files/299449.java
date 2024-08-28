public static <T> T newInstance(Class<T> type) {
        try {
            return type.newInstance();
        } catch (InstantiationException e) {
            throw new XOException("Cannot create instance of type '" + type.getName() + "'", e);
        } catch (IllegalAccessException e) {
            throw new XOException("Access denied to type '" + type.getName() + "'", e);
        }
    }