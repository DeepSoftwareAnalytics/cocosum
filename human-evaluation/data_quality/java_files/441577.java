public static final void setTimes(FileTime lastModifiedTime, FileTime lastAccessTime, FileTime createTime, Path... files) throws IOException
    {
        for (Path file : files)
        {
            BasicFileAttributeView view = Files.getFileAttributeView(file, BasicFileAttributeView.class);
            view.setTimes(lastModifiedTime, lastAccessTime, createTime);
        }
    }