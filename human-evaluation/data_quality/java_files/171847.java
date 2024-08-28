private void getEJBApplicationSubclasses(Set<Class<?>> classes, EJBEndpoint ejb, ClassLoader appClassloader) {
        final String methodName = "getEJBApplicationSubclasses";
        if (tc.isEntryEnabled()) {
            Tr.entry(tc, methodName);
        }

        if (classes == null) {
            if (tc.isEntryEnabled())
                Tr.exit(tc, methodName, Collections.emptySet());
            return;
        }

        Class<Application> appClass = Application.class;

        final String ejbClassName = ejb.getClassName();
        Class<?> c = null;
        try {
            c = appClassloader.loadClass(ejbClassName);
        } catch (ClassNotFoundException e) {

            if (tc.isDebugEnabled()) {
                Tr.debug(tc, methodName + " exit - due to Class Not Found for " + ejbClassName + ": " + e);
            }

        }

        if (c != null && appClass.isAssignableFrom(c)) {
            classes.add(c);
        }

        if (tc.isEntryEnabled())
            Tr.exit(tc, methodName, classes);

    }