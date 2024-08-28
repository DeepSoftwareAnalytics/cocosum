public <C extends Feature> C get(Class<C> feature)
    {
        final Feature found = typeToFeature.get(feature);
        if (found != null)
        {
            return feature.cast(found);
        }
        throw new LionEngineException(ERROR_FEATURE_NOT_FOUND + feature.getName());
    }