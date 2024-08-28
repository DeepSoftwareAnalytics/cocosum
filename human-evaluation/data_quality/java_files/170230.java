public boolean isToBeDeleted()
    {
        if (TraceComponent.isAnyTracingEnabled() && tc.isEntryEnabled())
        {
            SibTr.entry(tc, "isToBeDeleted");
            SibTr.exit(tc, "isToBeDeleted", Boolean.valueOf(toBeDeleted));
        }

        return toBeDeleted;
    }