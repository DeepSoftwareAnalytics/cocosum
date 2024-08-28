public void preAdd(Transaction transaction)
                    throws ObjectManagerException {
        final String methodName = "preAdd";
        if (Tracing.isAnyTracingEnabled() && trace.isEntryEnabled())
            trace.entry(this, cclass, methodName, new Object[] { transaction });

        // Whoever is calling us ought to hold a lock on the ManagedObject.
        testState(nextStateForAdd); // Test the state change.
        // We take a lock, only the owning transaction may now make non optimistic updates and 
        // all of those operations are now synchgronized on the transaction. 
        // TODO Should have a variant of lock that does not create a before image.
        lock(transaction.internalTransaction.getTransactionLock());
        // Bump the updateSequence to indicate the current version of the ManagedObject.
        // The caller is protected by the transaction lock.
        // The updateSequence is only meaningful while that lock is held.
        updateSequence++;

        if (Tracing.isAnyTracingEnabled() && trace.isEntryEnabled())
            trace.exit(this, cclass, methodName);
    }