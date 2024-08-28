public ApiResponse<List<CorporationMedalsResponse>> getCorporationsCorporationIdMedalsWithHttpInfo(
            Integer corporationId, String datasource, String ifNoneMatch, Integer page, String token)
            throws ApiException {
        com.squareup.okhttp.Call call = getCorporationsCorporationIdMedalsValidateBeforeCall(corporationId, datasource,
                ifNoneMatch, page, token, null);
        Type localVarReturnType = new TypeToken<List<CorporationMedalsResponse>>() {
        }.getType();
        return apiClient.execute(call, localVarReturnType);
    }