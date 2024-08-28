@Nonnull
  public FineUploaderBasic setMaxConnections (@Nonnegative final int nMaxConnections)
  {
    ValueEnforcer.isGT0 (nMaxConnections, "MaxConnections");
    m_nMaxConnections = nMaxConnections;
    return this;
  }