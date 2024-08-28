public Matrix4 setToTransform (IVector3 translation, IQuaternion rotation) {
        return setToRotation(rotation).setTranslation(translation);
    }