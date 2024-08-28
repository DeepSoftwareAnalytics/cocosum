public boolean add(final String member, final double score) {
        return doWithJedis(new JedisCallable<Boolean>() {
            @Override
            public Boolean call(Jedis jedis) {
                return jedis.zadd(getKey(), score, member) > 0;
            }
        });
    }