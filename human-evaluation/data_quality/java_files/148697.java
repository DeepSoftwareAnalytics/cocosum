protected void firePeerDiscovered(URI peerURI) {
        final NetworkServiceListener[] ilisteners;
        synchronized (this.listeners) {
            ilisteners = new NetworkServiceListener[this.listeners.size()];
            this.listeners.toArray(ilisteners);
        }
        for (final NetworkServiceListener listener : ilisteners) {
            listener.peerDiscovered(peerURI);
        }
    }