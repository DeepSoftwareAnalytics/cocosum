@Override
	public void cacheResult(List<CPRuleUserSegmentRel> cpRuleUserSegmentRels) {
		for (CPRuleUserSegmentRel cpRuleUserSegmentRel : cpRuleUserSegmentRels) {
			if (entityCache.getResult(
						CPRuleUserSegmentRelModelImpl.ENTITY_CACHE_ENABLED,
						CPRuleUserSegmentRelImpl.class,
						cpRuleUserSegmentRel.getPrimaryKey()) == null) {
				cacheResult(cpRuleUserSegmentRel);
			}
			else {
				cpRuleUserSegmentRel.resetOriginalValues();
			}
		}
	}