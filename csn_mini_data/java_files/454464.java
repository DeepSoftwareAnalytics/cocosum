private void readMoreChars() throws IOException {
    // Possibilities:
    // 1) array has space available on right hand side (between limit and capacity)
    // 2) array has space available on left hand side (before position)
    // 3) array has no space available
    //
    // In case 2 we shift the existing chars to the left, and in case 3 we create a bigger
    // array, then they both become case 1.

    if (availableCapacity(charBuffer) == 0) {
      if (charBuffer.position() > 0) {
        // (2) There is room in the buffer. Move existing bytes to the beginning.
        charBuffer.compact().flip();
      } else {
        // (3) Entire buffer is full, need bigger buffer.
        charBuffer = grow(charBuffer);
      }
    }

    // (1) Read more characters into free space at end of array.
    int limit = charBuffer.limit();
    int numChars = reader.read(charBuffer.array(), limit, availableCapacity(charBuffer));
    if (numChars == -1) {
      endOfInput = true;
    } else {
      charBuffer.limit(limit + numChars);
    }
  }