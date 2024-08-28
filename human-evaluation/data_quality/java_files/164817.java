@Override
	public Optional<Link> findLinkWithRel(LinkRelation rel, String representation) {

		return getLinks(representation).stream() //
				.filter(it -> it.hasRel(rel)) //
				.findFirst();
	}