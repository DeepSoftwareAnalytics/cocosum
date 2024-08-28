@Override
    public void writeContent(XMLExtendedStreamWriter writer, ModuleConfig value) throws XMLStreamException {

        writer.writeStartDocument();
        writer.writeStartElement(MODULE);
        writer.writeDefaultNamespace(MODULE_NS);

        if(moduleName == null) {
            throw new XMLStreamException("Module name is missing.");
        }
        writer.writeAttribute(NAME, moduleName);

        if (slotName != null) {
            writer.writeAttribute(SLOT, slotName);
        }

        if(properties != null) {
            writeNewLine(writer);
            writer.writeStartElement(PROPERTIES);
            for(Map.Entry<String, String> entry: properties.entrySet()) {
                writer.writeStartElement(PROPERTY);
                writer.writeAttribute(NAME, entry.getKey());
                writer.writeAttribute(VALUE, entry.getValue());
                writer.writeEndElement();
            }
            writer.writeEndElement();
        }

        if(mainClass != null) {
            writeNewLine(writer);
            writer.writeStartElement(MAIN_CLASS);
            writer.writeAttribute(NAME, mainClass);
            writer.writeEndElement();
        }

        if(resources != null) {
            writeNewLine(writer);
            writer.writeStartElement(RESOURCES);
            for(Resource res : resources) {
                res.writeContent(writer, res);
            }
            writer.writeEndElement();
        }

        if(dependencies != null) {
            writeNewLine(writer);
            writer.writeStartElement(DEPENDENCIES);
            for(Dependency dep : dependencies) {
                dep.writeContent(writer, dep);
            }
            writer.writeEndElement();
        }

        writeNewLine(writer);
        writer.writeEndElement();
        writer.writeEndDocument();
    }