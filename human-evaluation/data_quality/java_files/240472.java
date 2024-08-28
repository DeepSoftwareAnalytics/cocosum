protected void updateOrganization(Organization value, String xmlTag, Counter counter, Element element)
   {
      boolean shouldExist = value != null;
      Element root = updateElement(counter, element, xmlTag, shouldExist);
      if (shouldExist)
      {
         Counter innerCount = new Counter(counter.getDepth() + 1);
         findAndReplaceSimpleElement(innerCount, root, "name", value.getName(), null);
         findAndReplaceSimpleElement(innerCount, root, "url", value.getUrl(), null);
      }
   }