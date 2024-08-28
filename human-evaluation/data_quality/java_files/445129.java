public void warning(String format, Object... args)
    {
        if (isLoggable(WARNING))
        {
            logIt(WARNING, String.format(format, args));
        }
    }