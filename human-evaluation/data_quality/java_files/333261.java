public static LoadableImageData getImageDataFor(String ref) {
		LoadableImageData imageData;
		checkProperty();
		
		ref = ref.toLowerCase();
		
        if (ref.endsWith(".tga")) {
        	return new TGAImageData();
        } 
        if (ref.endsWith(".png")) {
        	CompositeImageData data = new CompositeImageData();
        	if (usePngLoader) {
        		data.add(new PNGImageData());
        	}
        	data.add(new ImageIOImageData());
        	
        	return data;
        } 
        
        return new ImageIOImageData();
	}