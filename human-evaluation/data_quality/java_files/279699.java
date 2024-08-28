private byte[] transform(byte[] classfileBuffer, InstrumentationActions instrumentationActions) {
		ClassReader classReader = new ClassReader(classfileBuffer);
		ClassWriter classWriter = new ClassWriter(classReader, ClassWriter.COMPUTE_FRAMES);
		ClassVisitor localVariableStateEmitterTestClassVisitor = new StateTrackingClassVisitor(classWriter,
				instrumentationActions);
		classReader.accept(localVariableStateEmitterTestClassVisitor, 0);
		return classWriter.toByteArray();
	}