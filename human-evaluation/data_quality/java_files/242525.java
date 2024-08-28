public static URL toURL(final String pURL) throws MalformedURLException {
    return new URL(null, pURL, sClasspathURLStreamHandler.supports(pURL) ? sClasspathURLStreamHandler : null);
  }