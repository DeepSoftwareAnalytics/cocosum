public static String addressToString(InetAddress addr) {
        if (addr == null) return "";

        StringBuilder hostnameAndAddress = new StringBuilder();
        String address = addr.getHostAddress();
        String hostnameOrAddress = ReverseDNSCache.hostnameOrAddress(addr);
        if (!hostnameOrAddress.equals(address)) {
            hostnameAndAddress.append(hostnameOrAddress);
        }
        hostnameAndAddress.append('/').append(address);
        return hostnameAndAddress.toString();
    }