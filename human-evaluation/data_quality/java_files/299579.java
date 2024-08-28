public static @NotNull String getTagWithComments(@NotNull String placeholder) {
    return "\n<!-- " + getPlaceholderName(placeholder) + " START -->\n"
        + placeholder
        + "\n<!-- " + getPlaceholderName(placeholder) + " END -->\n";
  }