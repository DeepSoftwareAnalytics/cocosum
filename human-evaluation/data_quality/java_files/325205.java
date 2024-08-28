public void writeBits(int b, int n) throws IOException {
		if (n <= capacity) {
			// all bits fit into the current buffer
			buffer = (buffer << n) | (b & (0xff >> (BITS_IN_BYTE - n)));
			capacity -= n;
			if (capacity == 0) {
				ostream.write(buffer);
				capacity = BITS_IN_BYTE;
				len++;
			}
		} else {
			// fill as many bits into buffer as possible
			buffer = (buffer << capacity)
					| ((b >>> (n - capacity)) & (0xff >> (BITS_IN_BYTE - capacity)));
			n -= capacity;
			ostream.write(buffer);
			len++;

			// possibly write whole bytes
			while (n >= 8) {
				n -= 8;
				ostream.write(b >>> n);
				len++;
			}

			// put the rest of bits into the buffer
			buffer = b; // Note: the high bits will be shifted out during
			// further filling
			capacity = BITS_IN_BYTE - n;
		}
	}