public void receiveAndProcess() {
        if (this.shouldContinue()) {
            this.receiver.receive(this.onReceiveHandler.getMaxEventCount())
                    .handleAsync(this.processAndReschedule, this.executor);
        } else {
            if (TRACE_LOGGER.isInfoEnabled()) {
                TRACE_LOGGER.info(String.format(Locale.US, "Stopping receive pump for eventHub (%s), consumerGroup (%s), partition (%s) as %s",
                        this.eventHubName, this.consumerGroupName, this.receiver.getPartitionId(),
                        this.stopPumpRaised.get() ? "per the request." : "pump ran into errors."));
            }

            this.stopPump.complete(null);
        }
    }