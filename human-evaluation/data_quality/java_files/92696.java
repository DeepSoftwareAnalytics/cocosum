public void invokeHookBoltFail(Tuple tuple, Duration failLatency) {
    if (taskHooks.size() != 0) {
      BoltFailInfo failInfo = new BoltFailInfo(tuple, getThisTaskId(), failLatency);

      for (ITaskHook taskHook : taskHooks) {
        taskHook.boltFail(failInfo);
      }
    }
  }