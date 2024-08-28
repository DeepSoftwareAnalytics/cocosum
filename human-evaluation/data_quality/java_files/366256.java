public static sslvserver_binding[] get(nitro_service service, String vservername[]) throws Exception{
		if (vservername !=null && vservername.length>0) {
			sslvserver_binding response[] = new sslvserver_binding[vservername.length];
			sslvserver_binding obj[] = new sslvserver_binding[vservername.length];
			for (int i=0;i<vservername.length;i++) {
				obj[i] = new sslvserver_binding();
				obj[i].set_vservername(vservername[i]);
				response[i] = (sslvserver_binding) obj[i].get_resource(service);
			}
			return response;
		}
		return null;
	}