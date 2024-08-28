private void startDraining(boolean overflow) {
    byteBuffer.flip();
    if (overflow && byteBuffer.remaining() == 0) {
      byteBuffer = ByteBuffer.allocate(byteBuffer.capacity() * 2);
    } else {
      draining = true;
    }
  }