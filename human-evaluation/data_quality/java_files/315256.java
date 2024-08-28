public static void puts(Object... messages) {

        for (Object message : messages) {
            IO.print(message);
            if (!(message instanceof Terminal.Escape)) IO.print(' ');
        }
        IO.println();

    }