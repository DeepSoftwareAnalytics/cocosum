public <O> ListenableFuture<O> borrowBatchAsync(int maxSize, Function<List<T>, BorrowResult<T, O>> function)
    {
        checkArgument(maxSize >= 0, "maxSize must be at least 0");

        ListenableFuture<List<T>> borrowedListFuture;
        synchronized (this) {
            List<T> list = getBatch(maxSize);
            if (!list.isEmpty()) {
                borrowedListFuture = immediateFuture(list);
                borrowerCount++;
            }
            else if (finishing && borrowerCount == 0) {
                borrowedListFuture = immediateFuture(ImmutableList.of());
            }
            else {
                borrowedListFuture = Futures.transform(
                        notEmptySignal,
                        ignored -> {
                            synchronized (this) {
                                List<T> batch = getBatch(maxSize);
                                if (!batch.isEmpty()) {
                                    borrowerCount++;
                                }
                                return batch;
                            }
                        },
                        executor);
            }
        }

        return Futures.transform(
                borrowedListFuture,
                elements -> {
                    // The borrowerCount field was only incremented for non-empty lists.
                    // Decrements should only happen for non-empty lists.
                    // When it should, it must always happen even if the caller-supplied function throws.
                    try {
                        BorrowResult<T, O> borrowResult = function.apply(elements);
                        if (elements.isEmpty()) {
                            checkArgument(borrowResult.getElementsToInsert().isEmpty(), "Function must not insert anything when no element is borrowed");
                            return borrowResult.getResult();
                        }
                        for (T element : borrowResult.getElementsToInsert()) {
                            offer(element);
                        }
                        return borrowResult.getResult();
                    }
                    finally {
                        if (!elements.isEmpty()) {
                            synchronized (this) {
                                borrowerCount--;
                                signalIfFinishing();
                            }
                        }
                    }
                }, directExecutor());
    }