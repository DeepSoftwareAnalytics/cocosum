public void close() {

        this.closed = true;

        List<CompletableFuture<?>> closeFutures = new ArrayList<>();
        while (hasConnections()) {
            for (StatefulRedisConnection<String, String> connection : drainConnections()) {
                closeFutures.add(connection.closeAsync());
            }
        }

        Futures.allOf(closeFutures).join();
    }