public static void addTableSRIDConstraint(Connection connection, TableLocation tableLocation, int srid)
            throws SQLException {
        //Alter table to set the SRID constraint
        if (srid > 0) {
            connection.createStatement().execute(String.format("ALTER TABLE %s ADD CHECK ST_SRID(the_geom)=%d",
                    tableLocation.toString(), srid));
        }
    }