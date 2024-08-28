public static ScopDomain guessScopDomain(String name, ScopDatabase scopDB) {
		List<ScopDomain> matches = new LinkedList<ScopDomain>();

		// Try exact match first
		ScopDomain domain = scopDB.getDomainByScopID(name);
		if (domain != null) {
			return domain;
		}

		// Didn't work. Guess it!
		logger.warn("Warning, could not find SCOP domain: " + name);

		Matcher scopMatch = scopPattern.matcher(name);
		if (scopMatch.matches()) {
			String pdbID = scopMatch.group(1);
			String chainName = scopMatch.group(2);
			String domainID = scopMatch.group(3);

			for (ScopDomain potentialSCOP : scopDB.getDomainsForPDB(pdbID)) {
				Matcher potMatch = scopPattern.matcher(potentialSCOP.getScopId());
				if (potMatch.matches()) {
					if (chainName.equals(potMatch.group(2)) || chainName.equals("_") || chainName.equals(".")
							|| potMatch.group(2).equals("_") || potMatch.group(2).equals(".")) {
						if (domainID.equals(potMatch.group(3)) || domainID.equals("_") || potMatch.group(3).equals("_")) {
							// Match, or near match
							matches.add(potentialSCOP);
						}
					}
				}
			}
		}

		Iterator<ScopDomain> match = matches.iterator();
		if (match.hasNext()) {
			ScopDomain bestMatch = match.next();
			if(logger.isWarnEnabled()) {
				StringBuilder warnMsg = new StringBuilder();
				warnMsg.append("Trying domain " + bestMatch.getScopId() + ".");
				if (match.hasNext()) {
					warnMsg.append(" Other possibilities: ");
					while (match.hasNext()) {
						warnMsg.append(match.next().getScopId()).append(" ");
					}
				}
				warnMsg.append(System.getProperty("line.separator"));
				logger.warn(warnMsg.toString());
			}
			return bestMatch;
		} else {
			return null;
		}
	}