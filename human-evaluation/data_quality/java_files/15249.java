public static AuthenticationResult authenticationResultFromRequest(final HttpServletRequest request)
  {
    final AuthenticationResult authenticationResult = (AuthenticationResult) request.getAttribute(
        AuthConfig.DRUID_AUTHENTICATION_RESULT
    );

    if (authenticationResult == null) {
      throw new ISE("Null authentication result");
    }

    return authenticationResult;
  }