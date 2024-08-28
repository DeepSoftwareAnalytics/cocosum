public static String join(Collection<?> objects, String separator) {
        String result = null;

        if (objects.size() > 0) {
            for (Object object : objects) {
                if (result == null) {
                    result = object.toString();
                } else {
                    result += separator + object.toString();
                }
            }
        } else {
            result = "";
        }

        return result;
    }