public boolean ceaseAllPublicFacingTrafficImmediately() {
        try {
            if (m_acceptor != null) {
                // This call seems to block until the shutdown is done
                // which is good becasue we assume there will be no new
                // connections afterward
                m_acceptor.shutdown();
            }
            if (m_adminAcceptor != null) {
                m_adminAcceptor.shutdown();
            }
        }
        catch (InterruptedException e) {
            // this whole method is really a best effort kind of thing...
            log.error(e);
            // if we didn't succeed, let the caller know and take action
            return false;
        }
        finally {
            m_isAcceptingConnections.set(false);
            // this feels like an unclean thing to do... but should work
            // for the purposes of cutting all responses right before we deliberately
            // end the process
            // m_cihm itself is thread-safe, and the regular shutdown code won't
            // care if it's empty... so... this.
            m_cihm.clear();
        }

        return true;
    }