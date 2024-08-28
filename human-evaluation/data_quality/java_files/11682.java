public boolean shouldSchedule(List<Action> actions) {
        List<ParametersAction> others = Util.filter(actions, ParametersAction.class);
        if (others.isEmpty()) {
            return !parameters.isEmpty();
        } else {
            // I don't think we need multiple ParametersActions, but let's be defensive
            Set<ParameterValue> params = new HashSet<>();
            for (ParametersAction other: others) {
                params.addAll(other.parameters);
            }
            return !params.equals(new HashSet<>(this.parameters));
        }
    }