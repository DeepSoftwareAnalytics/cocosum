public QueuedMessage[] getQueuedMessages(java.lang.Integer fromIndexInteger,java.lang.Integer toIndexInteger,java.lang.Integer totalMessagesPerpageInteger) throws Exception {
  
  int fromIndex=fromIndexInteger.intValue();
  int toIndex=toIndexInteger.intValue();
  int totalMessagesPerpage=totalMessagesPerpageInteger.intValue();
  
    if (TraceComponent.isAnyTracingEnabled() && tc.isEntryEnabled())
      SibTr.entry(tc, "getQueuedMessages  fromIndex="+fromIndex+" toIndex= "+toIndex+" totalMsgs= "+totalMessagesPerpage);

    List list = new ArrayList();

    Iterator iter = _c.getQueuedMessageIterator(fromIndex,toIndex,totalMessagesPerpage);//673411
	
    while (iter != null && iter.hasNext()) {
      SIMPQueuedMessageControllable o = (SIMPQueuedMessageControllable) iter.next();
      list.add(o);
    }

    List resultList = new ArrayList();
    iter = list.iterator();
    int i = 0;
    while (iter.hasNext()) {
      Object o = iter.next();
      QueuedMessage qm = SIBMBeanResultFactory.createSIBQueuedMessage((SIMPQueuedMessageControllable) o);
      resultList.add(qm);
    }

    QueuedMessage[] retValue = (QueuedMessage[])resultList.toArray(new QueuedMessage[0]);

    if (TraceComponent.isAnyTracingEnabled() && tc.isEntryEnabled())
      SibTr.exit(tc, "getQueuedMessagesfromIndex="+fromIndex+" toIndex= "+toIndex+" totalMsgs= "+totalMessagesPerpage, retValue);
    return retValue;
  }