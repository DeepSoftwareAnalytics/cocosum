@Deprecated
  @SuppressWarnings("unused")
  public void middleClick() {  // TODO(andreastt): Add this to Actions
    Point point = coordinates.inViewPort();
    exec.mouseAction(point.x, point.y, OperaMouseKeys.MIDDLE);
  }