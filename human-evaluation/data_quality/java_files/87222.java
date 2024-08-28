private HttpResponse executeUnparsed(boolean usingHead) throws IOException {
    HttpResponse response;
    if (uploader == null) {
      // normal request (not upload)
      response = buildHttpRequest(usingHead).execute();
    } else {
      // upload request
      GenericUrl httpRequestUrl = buildHttpRequestUrl();
      HttpRequest httpRequest = getAbstractGoogleClient()
          .getRequestFactory().buildRequest(requestMethod, httpRequestUrl, httpContent);
      boolean throwExceptionOnExecuteError = httpRequest.getThrowExceptionOnExecuteError();

      response = uploader.setInitiationHeaders(requestHeaders)
          .setDisableGZipContent(disableGZipContent).upload(httpRequestUrl);
      response.getRequest().setParser(getAbstractGoogleClient().getObjectParser());
      // process any error
      if (throwExceptionOnExecuteError && !response.isSuccessStatusCode()) {
        throw newExceptionOnError(response);
      }
    }
    // process response
    lastResponseHeaders = response.getHeaders();
    lastStatusCode = response.getStatusCode();
    lastStatusMessage = response.getStatusMessage();
    return response;
  }