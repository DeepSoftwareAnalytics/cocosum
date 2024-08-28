public Observable<ServiceResponse<SkuInfosInner>> listSkusWithServiceResponseAsync() {
        if (this.subscriptionId() == null) {
            throw new IllegalArgumentException("Parameter this.subscriptionId() is required and cannot be null.");
        }
        if (this.apiVersion() == null) {
            throw new IllegalArgumentException("Parameter this.apiVersion() is required and cannot be null.");
        }
        return service.listSkus(this.subscriptionId(), this.apiVersion(), this.acceptLanguage(), this.userAgent())
            .flatMap(new Func1<Response<ResponseBody>, Observable<ServiceResponse<SkuInfosInner>>>() {
                @Override
                public Observable<ServiceResponse<SkuInfosInner>> call(Response<ResponseBody> response) {
                    try {
                        ServiceResponse<SkuInfosInner> clientResponse = listSkusDelegate(response);
                        return Observable.just(clientResponse);
                    } catch (Throwable t) {
                        return Observable.error(t);
                    }
                }
            });
    }