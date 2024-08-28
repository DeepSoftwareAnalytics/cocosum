protected String replace(String template, String placeholder, String value)
    {
        if (template == null)
            return null;
        if ((placeholder == null) || (value == null))
            return template;

        while (true) {
            int index = template.indexOf(placeholder);
            if (index < 0)
                break;
            InternalStringBuilder temp = new InternalStringBuilder(template.substring(0, index));
            temp.append(value);
            temp.append(template.substring(index + placeholder.length()));
            template = temp.toString();
        }
        return template;
    }