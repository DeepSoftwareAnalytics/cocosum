public KeyAreaInfo setKeyArea(String strKeyName)
    {
        KeyAreaInfo keyArea = null;
        if (strKeyName == null)
            strKeyName = Constants.PRIMARY_KEY;
        for (m_iDefaultOrder = Constants.MAIN_KEY_AREA; m_iDefaultOrder < this.getKeyAreaCount() - Constants.MAIN_KEY_AREA; m_iDefaultOrder++)
        {
            keyArea = this.getKeyArea(m_iDefaultOrder);
            if (keyArea.getKeyName().equals(strKeyName))
                return keyArea;     // Found key area
        }
        if (Constants.PRIMARY_KEY.equals(strKeyName))
        {
            m_iDefaultOrder = Constants.MAIN_KEY_AREA;  // Set to default.
            return this.getKeyArea(m_iDefaultOrder);
        }
        m_iDefaultOrder = Constants.MAIN_KEY_AREA;  // Not found!!! Set to default.
        return null;
    }