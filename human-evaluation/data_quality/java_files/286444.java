private static String extractTemplateName(String filter) {
        Matcher matcher = TEMPLATE_FILTER_PATTERN.matcher(filter);
        if (matcher.matches()) {
            return matcher.group(1);
        } else {
            return "Unknown template";
        }
    }