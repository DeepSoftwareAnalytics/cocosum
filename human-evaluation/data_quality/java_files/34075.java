private boolean _validate(Token token, Term term) {

		if (token == null) {
			return true;
		}

		Set<String> terms = token.getTerms();
		if ((!terms.contains(Token.ALL)) && !terms.contains(term.getName()) && !(terms.contains(":" + term.getNatureStr()))) {
			return true;
		}

		boolean flag = token.getRegexs().size() != 0;

		for (String regex : token.getRegexs()) {
			if (term.getName().matches(regex)) {
				flag = false;
				break;
			}
		}

		return flag;
	}