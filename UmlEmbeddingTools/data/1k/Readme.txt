train:  codesearchnet data; data size = 1k
valid: codesearchnet data; data size = .5k
test:   codesearchnet data; data size = .5k
dataset.pkl : 
    dataset["ctrain"] ： train_method_summary
    dataset["cval"] ： val_method_summary
    dataset["ctest"] ： test_method_summary

    dataset["dtrain"] ： train_method_code
    dataset["dval"] ： val_method_code
    dataset["dtest"] ：test_method_code

   dataset["comstok"] = {"i2w": dict, "w2i": dict, "word_count": dict}

    # dataset["datstok"] = {"i2w": dict, "w2i": dict, "word_count": dict}
  
    # dataset["config"] = {"datvocabsize": int, "comvocabsize": int, "smlvocabsize": int, "datlen": int, "comlen": int,
    #                      "smllen": int}

we can use index of each sample (ctrain, cval, ctest )  to find the corresponding uml in train, valid and  test file.


