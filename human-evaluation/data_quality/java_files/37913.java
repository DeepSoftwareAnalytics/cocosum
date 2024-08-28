public static PluginProperties fromJarFile(final String filename) {
        final Properties properties = new Properties();
        try {
            final JarFile jarFile = new JarFile(requireNonNull(filename));
            final Optional<String> propertiesPath = getPropertiesPath(jarFile);

            if (propertiesPath.isPresent()) {
                LOG.debug("Loading <{}> from <{}>", propertiesPath.get(), filename);
                final ZipEntry entry = jarFile.getEntry(propertiesPath.get());

                if (entry != null) {
                    properties.load(jarFile.getInputStream(entry));
                } else {
                    LOG.debug("Plugin properties <{}> are missing in <{}>", propertiesPath.get(), filename);
                }
            }
        } catch (Exception e) {
            LOG.debug("Unable to load properties from plugin <{}>", filename, e);
        }

        return new PluginProperties(properties);
    }