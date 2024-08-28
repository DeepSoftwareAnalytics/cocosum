@Override
  public int read(byte[] b, int off, int len) throws IOException {
    // Obey InputStream contract.
    checkPositionIndexes(off, off + len, b.length);
    if (len == 0) {
      return 0;
    }

    // The rest of this method implements the process described by the CharsetEncoder javadoc.
    int totalBytesRead = 0;
    boolean doneEncoding = endOfInput;

    DRAINING:
    while (true) {
      // We stay in draining mode until there are no bytes left in the output buffer. Then we go
      // back to encoding/flushing.
      if (draining) {
        totalBytesRead += drain(b, off + totalBytesRead, len - totalBytesRead);
        if (totalBytesRead == len || doneFlushing) {
          return (totalBytesRead > 0) ? totalBytesRead : -1;
        }
        draining = false;
        byteBuffer.clear();
      }

      while (true) {
        // We call encode until there is no more input. The last call to encode will have endOfInput
        // == true. Then there is a final call to flush.
        CoderResult result;
        if (doneFlushing) {
          result = CoderResult.UNDERFLOW;
        } else if (doneEncoding) {
          result = encoder.flush(byteBuffer);
        } else {
          result = encoder.encode(charBuffer, byteBuffer, endOfInput);
        }

        if (result.isOverflow()) {
          // Not enough room in output buffer--drain it, creating a bigger buffer if necessary.
          startDraining(true);
          continue DRAINING;
        } else if (result.isUnderflow()) {
          // If encoder underflows, it means either:
          // a) the final flush() succeeded; next drain (then done)
          // b) we encoded all of the input; next flush
          // c) we ran of out input to encode; next read more input
          if (doneEncoding) { // (a)
            doneFlushing = true;
            startDraining(false);
            continue DRAINING;
          } else if (endOfInput) { // (b)
            doneEncoding = true;
          } else { // (c)
            readMoreChars();
          }
        } else if (result.isError()) {
          // Only reach here if a CharsetEncoder with non-REPLACE settings is used.
          result.throwException();
          return 0; // Not called.
        }
      }
    }
  }