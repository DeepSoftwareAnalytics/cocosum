public static <S> void removeSolutionsFromList(List<S> solutionList, int numberOfSolutionsToRemove) {
    if (solutionList.size() < numberOfSolutionsToRemove) {
      throw new JMetalException("The list size (" + solutionList.size() + ") is lower than " +
          "the number of solutions to remove (" + numberOfSolutionsToRemove + ")");
    }

    for (int i = 0; i < numberOfSolutionsToRemove; i++) {
      solutionList.remove(0);
    }
  }