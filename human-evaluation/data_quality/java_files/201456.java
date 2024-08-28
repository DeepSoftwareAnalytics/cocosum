private void removeTracker(TaskTracker tracker) {
    String trackerName = tracker.getTrackerName();
    // Remove completely after marking the tasks as 'KILLED'
    lostTaskTracker(tracker);
    TaskTrackerStatus status = tracker.getStatus();

    // tracker is lost, and if it is blacklisted, remove
    // it from the count of blacklisted trackers in the cluster
    if (isBlacklisted(trackerName)) {
     faultyTrackers.decrBlackListedTrackers(1);
    }
    updateTaskTrackerStatus(trackerName, null);
    statistics.taskTrackerRemoved(trackerName);
    getInstrumentation().decTrackers(1);
  }