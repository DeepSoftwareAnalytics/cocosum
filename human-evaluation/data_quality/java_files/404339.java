private void handleUnready(String address) {
    log.debug(String.format("%s - Received unready message from: %s", NetworkManager.this, address));
    ready.remove(address);
    checkUnready();
  }