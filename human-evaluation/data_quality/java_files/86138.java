private void performPostCompilationTasksInternal() {
    if (options.devMode == DevMode.START_AND_END) {
      runValidityCheck();
    }
    setProgress(1.0, "recordFunctionInformation");

    if (tracker != null) {
      tracker.outputTracerReport();
    }
  }