public static nsspparams get(nitro_service service) throws Exception{
		nsspparams obj = new nsspparams();
		nsspparams[] response = (nsspparams[])obj.get_resources(service);
		return response[0];
	}