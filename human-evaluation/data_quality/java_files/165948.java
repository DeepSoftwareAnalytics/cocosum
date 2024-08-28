public CreateMBean readCreateMBean(InputStream in) throws ConversionException, IOException, ClassNotFoundException {
        JSONObject json = parseObject(in);
        CreateMBean ret = new CreateMBean();
        ret.objectName = readObjectName(json.get(N_OBJECTNAME));
        ret.className = readStringInternal(json.get(N_CLASSNAME));
        ret.loaderName = readObjectName(json.get(N_LOADERNAME));
        ret.params = readPOJOArray(json.get(N_PARAMS));
        ret.signature = readStringArrayInternal(json.get(N_SIGNATURE));
        ret.useLoader = readBooleanInternal(json.get(N_USELOADER));
        ret.useSignature = readBooleanInternal(json.get(N_USESIGNATURE));
        return ret;
    }