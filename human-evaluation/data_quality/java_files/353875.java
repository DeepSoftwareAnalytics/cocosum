public void setFullUrl(String fullUrl) {
        if (fullUrl == null) {
            return;
        }

        Uri parsedUri = Uri.parse(fullUrl);

        if (!WonderPushUriHelper.isAPIUri(parsedUri)) {
            mWebView.loadUrl(fullUrl);
        } else {
            setResource(WonderPushUriHelper.getResource(parsedUri), WonderPushUriHelper.getParams(parsedUri));
        }
    }