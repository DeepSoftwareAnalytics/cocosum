private void insert(final char[] revB, final DiffBlock curB)
		throws UnsupportedEncodingException
	{

		String text = copy(revB, curB.getRevBStart(), curB.getRevBEnd());

		// Insert (C S L T)
		DiffPart action = new DiffPart(DiffAction.INSERT);

		// S
		action.setStart(version.length());
		codecData.checkBlocksizeS(version.length());

		// L T
		action.setText(text);
		codecData.checkBlocksizeL(text.getBytes(WIKIPEDIA_ENCODING).length);

		diff.add(action);

		version.append(text);
	}