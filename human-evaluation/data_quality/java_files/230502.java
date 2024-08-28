private String dumpZoneNAryDetails(StoreRoutingPlan storeRoutingPlan) {
        StringBuilder sb = new StringBuilder();

        sb.append("\tDetailed Dump (Zone N-Aries):").append(Utils.NEWLINE);
        for(Node node: storeRoutingPlan.getCluster().getNodes()) {
            int zoneId = node.getZoneId();
            int nodeId = node.getId();
            sb.append("\tNode ID: " + nodeId + " in zone " + zoneId).append(Utils.NEWLINE);
            List<Integer> naries = storeRoutingPlan.getZoneNAryPartitionIds(nodeId);
            Map<Integer, List<Integer>> zoneNaryTypeToPartitionIds = new HashMap<Integer, List<Integer>>();
            for(int nary: naries) {
                int zoneReplicaType = storeRoutingPlan.getZoneNaryForNodesPartition(zoneId,
                                                                                    nodeId,
                                                                                    nary);
                if(!zoneNaryTypeToPartitionIds.containsKey(zoneReplicaType)) {
                    zoneNaryTypeToPartitionIds.put(zoneReplicaType, new ArrayList<Integer>());
                }
                zoneNaryTypeToPartitionIds.get(zoneReplicaType).add(nary);
            }

            for(int replicaType: new TreeSet<Integer>(zoneNaryTypeToPartitionIds.keySet())) {
                sb.append("\t\t" + replicaType + " : ");
                sb.append(zoneNaryTypeToPartitionIds.get(replicaType).toString());
                sb.append(Utils.NEWLINE);
            }
        }

        return sb.toString();
    }