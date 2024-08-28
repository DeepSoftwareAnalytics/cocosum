public Waiter<DescribeDBSnapshotsRequest> dBSnapshotAvailable() {

        return new WaiterBuilder<DescribeDBSnapshotsRequest, DescribeDBSnapshotsResult>()
                .withSdkFunction(new DescribeDBSnapshotsFunction(client))
                .withAcceptors(new DBSnapshotAvailable.IsAvailableMatcher(), new DBSnapshotAvailable.IsDeletedMatcher(),
                        new DBSnapshotAvailable.IsDeletingMatcher(), new DBSnapshotAvailable.IsFailedMatcher(),
                        new DBSnapshotAvailable.IsIncompatiblerestoreMatcher(), new DBSnapshotAvailable.IsIncompatibleparametersMatcher())
                .withDefaultPollingStrategy(new PollingStrategy(new MaxAttemptsRetryStrategy(60), new FixedDelayStrategy(30)))
                .withExecutorService(executorService).build();
    }