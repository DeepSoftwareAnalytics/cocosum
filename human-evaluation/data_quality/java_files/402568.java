@Override
    public int doEndTag() throws JspException {
        Locale locale;

        if (value instanceof Locale) {
            locale = (Locale) value;
        } else if (value instanceof String && !"".equals(((String)value).trim())) {
            locale = LocaleUtil.parseLocale((String) value, variant);
        } else {
            locale = Locale.getDefault();
        }

        Config.set(pageContext, Config.FMT_LOCALE, locale, scope);
        setResponseLocale(pageContext, locale);

        return EVAL_PAGE;
    }