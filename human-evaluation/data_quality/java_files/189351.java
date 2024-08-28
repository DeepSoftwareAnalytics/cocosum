public static HijrahDate of(int prolepticYear, int monthOfYear, int dayOfMonth) {
        return (prolepticYear >= 1) ?
            HijrahDate.of(HijrahEra.AH, prolepticYear, monthOfYear, dayOfMonth) :
            HijrahDate.of(HijrahEra.BEFORE_AH, 1 - prolepticYear, monthOfYear, dayOfMonth);
    }