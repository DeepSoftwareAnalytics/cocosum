@Override
	public List<Diagnostic> getDiagnostics(final int severity) {
		InputModel model = getComponentModel();

		switch (severity) {
			case Diagnostic.ERROR:
				return model.errorDiagnostics;
			case Diagnostic.WARNING:
				return model.warningDiagnostics;
			case Diagnostic.INFO:
				return model.infoDiagnostics;
			case Diagnostic.SUCCESS:
				return model.successDiagnostics;
			default:
				return null;
		}
	}