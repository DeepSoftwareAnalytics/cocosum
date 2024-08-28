public void readDashboardRolesConfig(@Observes @Initialized(ApplicationScoped.class) ServletContext sc) {
		readFromConfigurationRoles();
		readFromInitParameter(sc);
		logger.debug("'{}' value : '{}'.", Constants.Options.DASHBOARD_ROLES, roles);
	}