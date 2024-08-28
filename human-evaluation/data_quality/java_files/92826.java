void appendToBootstrapClassLoaderSearch(InputStream jarStream) throws IOException {
        File jarFile = File.createTempFile("mockito-boot", ".jar");
        jarFile.deleteOnExit();

        byte[] buffer = new byte[64 * 1024];
        try (OutputStream os = new FileOutputStream(jarFile)) {
            while (true) {
                int numRead = jarStream.read(buffer);
                if (numRead == -1) {
                    break;
                }

                os.write(buffer, 0, numRead);
            }
        }

        nativeAppendToBootstrapClassLoaderSearch(jarFile.getAbsolutePath());
    }