@Override
	public void encodeBegin(FacesContext context) throws IOException {

		// Initialize attributes
		this.useRandomParam = (String) getAttributes().get("useRandomParam");

		String src = (String) getAttributes().get("src");

		ResponseWriter writer = context.getResponseWriter();
		HttpServletRequest request = ((HttpServletRequest) context.getExternalContext().getRequest());

		ResourceBundlesHandler bundler = getResourceBundlesHandler(context);

		// src is mandatory
		if (null == src)
			throw new IllegalStateException("The src attribute is mandatory for this Jawr tag. ");

		// Refresh the config if needed
		if (RendererRequestUtils.refreshConfigIfNeeded(request, bundler.getConfig())) {
			bundler = getResourceBundlesHandler(context);
		}

		// Get an instance of the renderer.
		if (null == this.renderer || !this.renderer.getBundler().getConfig().isValid())
			this.renderer = createRenderer(context);

		RendererRequestUtils.setRequestDebuggable(request, renderer.getBundler().getConfig());
		try {
			BundleRendererContext ctx = RendererRequestUtils.getBundleRendererContext(request, renderer);
			renderer.renderBundleLinks(src, ctx, writer);
		} finally {
			// Reset the Thread local for the Jawr context
			ThreadLocalJawrContext.reset();
		}
		super.encodeBegin(context);

	}