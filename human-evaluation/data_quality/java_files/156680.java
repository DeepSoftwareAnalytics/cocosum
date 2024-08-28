public void updateAckMailboxes(int partition, Set<Long> newHSIds) {
        ImmutableList<Long> replicaHSIds = m_replicasHSIds.get(partition);
        synchronized (m_dataSourcesByPartition) {
            Map<String, ExportDataSource> partitionMap = m_dataSourcesByPartition.get(partition);
            if (partitionMap == null) {
                return;
            }
            for( ExportDataSource eds: partitionMap.values()) {
                eds.updateAckMailboxes(Pair.of(m_mbox, replicaHSIds));
                if (newHSIds != null && !newHSIds.isEmpty()) {
                    // In case of newly joined or rejoined streams miss any RELEASE_BUFFER event,
                    // master stream resends the event when the export mailbox is aware of new streams.
                    eds.forwardAckToNewJoinedReplicas(newHSIds);
                    // After rejoin, new data source may contain the data which current master doesn't have,
                    //  only on master stream if it is blocked by the gap
                    eds.queryForBestCandidate();
                }
            }
        }
    }