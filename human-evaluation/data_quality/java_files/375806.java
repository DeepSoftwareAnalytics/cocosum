public ApiResponse authorizeAppUser(String email, String password) {
        validateNonEmptyParam(email, "email");
        validateNonEmptyParam(password,"password");
        assertValidApplicationId();
        loggedInUser = null;
        accessToken = null;
        currentOrganization = null;
        Map<String, Object> formData = new HashMap<String, Object>();
        formData.put("grant_type", "password");
        formData.put("username", email);
        formData.put("password", password);
        ApiResponse response = apiRequest(HttpMethod.POST, formData, null,
                organizationId, applicationId, "token");
        if (response == null) {
            return response;
        }
        if (!isEmpty(response.getAccessToken()) && (response.getUser() != null)) {
            loggedInUser = response.getUser();
            accessToken = response.getAccessToken();
            currentOrganization = null;
            log.info("Client.authorizeAppUser(): Access token: " + accessToken);
        } else {
            log.info("Client.authorizeAppUser(): Response: " + response);
        }
        return response;
    }