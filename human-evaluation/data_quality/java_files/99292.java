protected void reportItem(OPFItem item)
  {
    if (item.isInSpine())
    {
      report.info(item.getPath(), FeatureEnum.IS_SPINEITEM, "true");
      report.info(item.getPath(), FeatureEnum.IS_LINEAR, String.valueOf(item.isLinear()));
    }
    if (item.isNcx())
    {
      report.info(item.getPath(), FeatureEnum.HAS_NCX, "true");
      if (!item.getMimeType().equals("application/x-dtbncx+xml"))
      {
        report.message(MessageId.OPF_050,
            EPUBLocation.create(path, item.getLineNumber(), item.getColumnNumber()));
      }
    }
  }