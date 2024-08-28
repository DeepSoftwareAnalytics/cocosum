@Nonnull
  public final XMLWriterSettings setNamespaceContext (@Nullable final INamespaceContext aNamespaceContext)
  {
    // A namespace context must always be present, to resolve default namespaces
    m_aNamespaceContext = aNamespaceContext != null ? aNamespaceContext : new MapBasedNamespaceContext ();
    return this;
  }