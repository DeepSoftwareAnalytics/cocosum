private void overrideUseMavenPatterns(T overrider, Class overriderClass) {
        if (useMavenPatternsOverrideList.contains(overriderClass.getSimpleName())) {
            try {
                Field useMavenPatternsField = overriderClass.getDeclaredField("useMavenPatterns");
                useMavenPatternsField.setAccessible(true);
                Object useMavenPatterns = useMavenPatternsField.get(overrider);

                if (useMavenPatterns == null) {
                    Field notM2CompatibleField = overriderClass.getDeclaredField("notM2Compatible");
                    notM2CompatibleField.setAccessible(true);
                    Object notM2Compatible = notM2CompatibleField.get(overrider);
                    if (notM2Compatible instanceof Boolean && notM2Compatible != null) {
                        useMavenPatternsField.set(overrider, !(Boolean)notM2Compatible);
                    }
                }
            } catch (NoSuchFieldException | IllegalAccessException e) {
                converterErrors.add(getConversionErrorMessage(overrider, e));
            }
        }
    }