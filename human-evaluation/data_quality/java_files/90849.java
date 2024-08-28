private byte[] collectProcessResultsBytes(Process process, ProcessBuilder builder,
      SubProcessIOFiles outPutFiles) throws Exception {

    Byte[] results;

    try {

      LOG.debug(String.format("Executing process %s", createLogEntryFromInputs(builder.command())));

      // If process exit value is not 0 then subprocess failed, record logs
      if (process.exitValue() != 0) {
        outPutFiles.copyOutPutFilesToBucket(configuration, FileUtils.toStringParams(builder));
        String log = createLogEntryForProcessFailure(process, builder.command(), outPutFiles);
        throw new Exception(log);
      }

      // If no return file then either something went wrong or the binary is setup incorrectly for
      // the ret file either way throw error
      if (!Files.exists(outPutFiles.resultFile)) {
        String log = createLogEntryForProcessFailure(process, builder.command(), outPutFiles);
        outPutFiles.copyOutPutFilesToBucket(configuration, FileUtils.toStringParams(builder));
        throw new Exception(log);
      }

      // Everything looks healthy return bytes
      return Files.readAllBytes(outPutFiles.resultFile);

    } catch (Exception ex) {
      String log = String.format("Unexpected error runnng process. %s error message was %s",
          createLogEntryFromInputs(builder.command()), ex.getMessage());
      throw new Exception(log);
    }
  }