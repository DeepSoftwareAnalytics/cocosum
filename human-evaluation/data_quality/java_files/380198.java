public void execute() throws MojoExecutionException {
		
		
		getLog().info("Start, BS: " + System.getProperty("os.name"));
		if (System.getProperty("os.name").contains("indows")){
			CPPATHSEPERATOR = ";";
			PATHSEPERATOR="\\";
		}else{
			CPPATHSEPERATOR=":";
			PATHSEPERATOR="/";
		}
		System.out.println("Execute: " + standardoutput);
		tests = 0;
		failure = 0;
		error = 0;
		
		if (standardoutput != null){
			File output = new File(standardoutput);
			try {
				bw = new BufferedWriter( new FileWriter(output));
			} catch (IOException e1) {
				// TODO Automatisch generierter Erfassungsblock
				e1.printStackTrace();
			}
		}
		else{
			File output = new File("stdout.txt");
			try {
				bw = new BufferedWriter( new FileWriter(output));
			} catch (IOException e1) {
				// TODO Automatisch generierter Erfassungsblock
				e1.printStackTrace();
			}
		}
		
		File dir = new File("src/test/java/");
		FileFilter fileFilter = new RegexFileFilter("[A-z/]*.java$");

		project = project.getExecutionProject();
		runTestFile(dir, fileFilter);

		getLog().info("Tests: " + tests + " Fehlschläge: " + failure
				+ " Fehler: " + error);

	}