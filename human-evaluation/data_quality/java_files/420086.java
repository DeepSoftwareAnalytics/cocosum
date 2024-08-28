public static void options(String url, HttpConsumer<HttpExchange> endpoint, MediaTypes... mediaTypes) {
        addResource(Methods.OPTIONS, url, endpoint, mediaTypes);
    }