public static void closeStatement( Statement st, Logger logger ) {

		try {
			if( st != null )
				st.close();

		} catch( SQLException e ) {
			// Not important.
			Utils.logException( logger, e );
		}
	}