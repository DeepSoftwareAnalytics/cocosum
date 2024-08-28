@Override
    public PackageDoc containingPackage() {
        PackageDocImpl p = env.getPackageDoc(tsym.packge());
        if (p.setDocPath == false) {
            FileObject docPath;
            try {
                Location location = env.fileManager.hasLocation(StandardLocation.SOURCE_PATH)
                    ? StandardLocation.SOURCE_PATH : StandardLocation.CLASS_PATH;

                docPath = env.fileManager.getFileForInput(
                        location, p.qualifiedName(), "package.html");
            } catch (IOException e) {
                docPath = null;
            }

            if (docPath == null) {
                // fall back on older semantics of looking in same directory as
                // source file for this class
                SourcePosition po = position();
                if (env.fileManager instanceof StandardJavaFileManager &&
                        po instanceof SourcePositionImpl) {
                    URI uri = ((SourcePositionImpl) po).filename.toUri();
                    if ("file".equals(uri.getScheme())) {
                        File f = new File(uri);
                        File dir = f.getParentFile();
                        if (dir != null) {
                            File pf = new File(dir, "package.html");
                            if (pf.exists()) {
                                StandardJavaFileManager sfm = (StandardJavaFileManager) env.fileManager;
                                docPath = sfm.getJavaFileObjects(pf).iterator().next();
                            }
                        }

                    }
                }
            }

            p.setDocPath(docPath);
        }
        return p;
    }