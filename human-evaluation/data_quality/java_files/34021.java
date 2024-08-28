public String queryParamOrDefault(String queryParam, String defaultValue) {
        String value = queryParams(queryParam);
        return value != null ? value : defaultValue;
    }