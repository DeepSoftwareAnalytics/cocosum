public boolean showElement(String key, String defaultValue, Properties displayOptions) {

        if (defaultValue == null) {
            return ((displayOptions != null) && Boolean.valueOf(displayOptions.getProperty(key)).booleanValue());
        }
        if (displayOptions == null) {
            return Boolean.valueOf(defaultValue).booleanValue();
        }
        return Boolean.valueOf(displayOptions.getProperty(key, defaultValue)).booleanValue();
    }