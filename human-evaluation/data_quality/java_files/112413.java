protected void fireStartDoc()
        throws org.xml.sax.SAXException
    {
        if (m_tracer != null)
        {
            flushMyWriter();
            m_tracer.fireGenerateEvent(SerializerTrace.EVENTTYPE_STARTDOCUMENT);
        }     	    
    }