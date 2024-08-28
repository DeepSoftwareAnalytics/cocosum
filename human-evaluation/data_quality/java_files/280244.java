public boolean setView(final Category category) {
    LOGGER.trace("CategoryController, setView: " + category);
    CategoryView categoryView = views.get(category);
    if (categoryView != null) { // view is loaded
      setContent(categoryView);
      // Binding for ScrollPane
      categoryView.minWidthProperty().bind(widthProperty().subtract(SCROLLBAR_SUBTRACT));
      displayedCategoryView.setValue(categoryView);
      displayedCategoryPresenter.setValue(getPresenter(category));
      return true;
    } else {
      LOGGER.info("Category " + category.getDescription() + " hasn't been loaded!");
      return false;
    }
  }