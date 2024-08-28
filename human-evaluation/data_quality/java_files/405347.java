public static int compare (Comparable<?> c1, Comparable<?> c2)
    {
        @SuppressWarnings("unchecked") Comparable<Object> cc1 = (Comparable<Object>)c1;
        @SuppressWarnings("unchecked") Comparable<Object> cc2 = (Comparable<Object>)c2;
        return cc1.compareTo(cc2);
    }