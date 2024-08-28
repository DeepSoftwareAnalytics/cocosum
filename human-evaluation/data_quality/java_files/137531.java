public static CPMeasurementUnit fetchByUuid_First(String uuid,
		OrderByComparator<CPMeasurementUnit> orderByComparator) {
		return getPersistence().fetchByUuid_First(uuid, orderByComparator);
	}