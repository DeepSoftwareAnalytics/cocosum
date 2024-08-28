public void setMaterial(GVRMaterial material)
    {
        mMaterial = material;
        NativeRenderPass.setMaterial(getNative(), material.getNative());
    }