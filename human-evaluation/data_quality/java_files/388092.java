public ReportsConfig getReportsConfig(Collection<String> restrictToNamed) {
        resetErrors();
        ReportsConfig result = new ReportsConfig();
        InternalType[] reports = getReportList();
        if (reports == null)
        {
            errors.add("No reports found.");
            return result;
        }
        for (InternalType report : reports)
        {
            String name = "Unknown";
            final String named = reportToName(report);
            if (restrictToNamed != null && !restrictToNamed.contains(named)) continue;
            try {
                name = loadReport(result, report);
            } catch (Exception e) {
                errors.add("Error in report " + named + ": " + e.getMessage());
                e.printStackTrace();
                continue;
            }
        }
        return result;
    }