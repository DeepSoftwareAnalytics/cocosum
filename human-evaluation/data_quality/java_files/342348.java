private Content getTreeForClassHelper(TypeMirror type) {
        Content li = new HtmlTree(HtmlTag.LI);
        if (type.equals(typeElement.asType())) {
            Content typeParameters = getTypeParameterLinks(
                    new LinkInfoImpl(configuration, LinkInfoImpl.Kind.TREE,
                    typeElement));
            if (configuration.shouldExcludeQualifier(utils.containingPackage(typeElement).toString())) {
                li.addContent(utils.asTypeElement(type).getSimpleName());
                li.addContent(typeParameters);
            } else {
                li.addContent(utils.asTypeElement(type).getQualifiedName());
                li.addContent(typeParameters);
            }
        } else {
            Content link = getLink(new LinkInfoImpl(configuration,
                    LinkInfoImpl.Kind.CLASS_TREE_PARENT, type)
                    .label(configuration.getClassName(utils.asTypeElement(type))));
            li.addContent(link);
        }
        return li;
    }