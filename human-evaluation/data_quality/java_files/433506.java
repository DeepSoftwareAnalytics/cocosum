public static MemcachedManagerBuilder memcacheAddOn() {
        final String memcacheServers = System.getenv("MEMCACHE_SERVERS");
        return memcachedConfig()
                .username(System.getenv("MEMCACHE_USERNAME"))
                .password(System.getenv("MEMCACHE_PASSWORD"))
                .url(memcacheServers == null ? DEFAULT_URL : memcacheServers);
    }