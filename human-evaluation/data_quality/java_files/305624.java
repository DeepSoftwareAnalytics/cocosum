@Override
   public void validate() throws ValidateException
   {
      if (userName != null && securityDomain != null)
      {
         throw new ValidateException(bundle.invalidSecurityConfiguration());
      }
   }