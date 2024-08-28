protected void get_subwindow( T image , GrayF64 output ) {

		// copy the target region

		interp.setImage(image);
		int index = 0;
		for( int y = 0; y < workRegionSize; y++ ) {
			float yy = regionTrack.y0 + y*stepY;

			for( int x = 0; x < workRegionSize; x++ ) {
				float xx = regionTrack.x0 + x*stepX;

				if( interp.isInFastBounds(xx,yy))
					output.data[index++] = interp.get_fast(xx,yy);
				else if( BoofMiscOps.checkInside(image, xx, yy))
					output.data[index++] = interp.get(xx, yy);
				else {
					// randomize to make pixels outside the image poorly correlate.  It will then focus on matching
					// what's inside the image since it has structure
					output.data[index++] = rand.nextFloat()*maxPixelValue;
				}
			}
		}

		// normalize values to be from -0.5 to 0.5
		PixelMath.divide(output, maxPixelValue, output);
		PixelMath.plus(output, -0.5f, output);
		// apply the cosine window to it
		PixelMath.multiply(output,cosine,output);
	}