public static final ByteBuffer getImageByteBuffer(Webcam webcam, String format) {
		return ByteBuffer.wrap(getImageBytes(webcam, format));
	}