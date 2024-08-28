public synchronized void rollback(
            Savepoint savepoint) throws SQLException {

        JDBCSavepoint sp;

        checkClosed();

        if (savepoint == null) {
            throw Util.nullArgument();
        }

        if (!(savepoint instanceof JDBCSavepoint)) {
            String msg = Error.getMessage(ErrorCode.X_3B001);

            throw Util.invalidArgument(msg);
        }
        sp = (JDBCSavepoint) savepoint;

        if (JDBCDatabaseMetaData.JDBC_MAJOR >= 4 && sp.name == null) {
            String msg = Error.getMessage(ErrorCode.X_3B001);

            throw Util.invalidArgument(msg);
        }

        if (this != sp.connection) {
            String msg = Error.getMessage(ErrorCode.X_3B001);

            throw Util.invalidArgument(msg);
        }

        if (JDBCDatabaseMetaData.JDBC_MAJOR >= 4 && getAutoCommit()) {
            sp.name       = null;
            sp.connection = null;

            throw Util.sqlException(ErrorCode.X_3B001);
        }

        try {
            sessionProxy.rollbackToSavepoint(sp.name);

            if (JDBCDatabaseMetaData.JDBC_MAJOR >= 4) {
                sp.connection = null;
                sp.name       = null;
            }
        } catch (HsqlException e) {
            throw Util.sqlException(e);
        }
    }