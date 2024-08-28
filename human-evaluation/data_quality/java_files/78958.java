public void printTo(Writer out, ReadablePeriod period) throws IOException {
        checkPrinter();
        checkPeriod(period);
        
        getPrinter().printTo(out, period, iLocale);
    }