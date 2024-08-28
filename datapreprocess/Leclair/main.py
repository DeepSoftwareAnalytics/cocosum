import pickle
import tokenizer

"""
Remove the dependence of tokenizer class and Keras library for original Leclair dataset
"""

if __name__ == '__main__':

    # sbt_data = True
    # dataprep = "/datadrive/CodeSummary_Data/raw_data/funcom/data/sbt"
    # output_path = "./sbt_data.pkl"

    sbt_data = False
    dataprep = "/datadrive/CodeSummary_Data/raw_data/funcom/data/standard"
    output_path = "./standard_data.pkl"

    print("Load tdatstok: %s/dats.tok" % (dataprep))
    with open('%s/dats.tok' % (dataprep), 'rb') as input:
        tdatstok = pickle.load(input, encoding='UTF-8')

    print("Load sdatstok: %s/dats.tok" % (dataprep))
    with open('%s/dats.tok' % (dataprep), 'rb') as input:
        sdatstok = pickle.load(input, encoding='UTF-8')

    print("Load comstok: %s/coms.tok" % (dataprep))
    with open('%s/coms.tok' % (dataprep), 'rb') as input:
        comstok = pickle.load(input, encoding='UTF-8')

    if not sbt_data:
        print("Load smltok: %s/smls.tok" % (dataprep))
        with open('%s/smls.tok' % (dataprep), 'rb') as input:
            smltok = pickle.load(input, encoding='UTF-8')

    print("Load seqdata: %s/dataset.pkl" % (dataprep))
    with open('%s/dataset.pkl' % (dataprep), 'rb') as input:
        seqdata = pickle.load(input)

    dataset = {}

    # dict = {key:ndarray}
    dataset["ctrain"] = seqdata["ctrain"]
    dataset["cval"] = seqdata["cval"]
    dataset["ctest"] = seqdata["ctest"]

    dataset["dtrain"] = seqdata["dtrain"]
    dataset["dval"] = seqdata["dval"]
    dataset["dtest"] = seqdata["dtest"]

    if not sbt_data:
        dataset["strain"] = seqdata["strain"]
        dataset["sval"] = seqdata["sval"]
        dataset["stest"] = seqdata["stest"]

    # i2w: id to token mapping. 1: <s>, 2: </s>, ..., max_id: oov_index. 0 is padding token but it is not contained in the dict

    # Method Comment Tokens
    # dataset["comstok"] = {"i2w": dict, "w2i": dict, "word_count": dict}
    dataset["comstok"] = {"i2w": seqdata["comstok"].i2w, "w2i": seqdata["comstok"].w2i, "word_count": dict(seqdata["comstok"].word_count)}
    # Method Code Tokens
    # dataset["datstok"] = {"i2w": dict, "w2i": dict, "word_count": dict}
    dataset["datstok"] = {"i2w": seqdata["datstok"].i2w, "w2i": seqdata["datstok"].w2i, "word_count": dict(seqdata["datstok"].word_count)}

    if not sbt_data:
        # Tree Tokens
        # dataset["smltok"] = {"i2w": dict, "w2i": dict, "word_count": dict}
        dataset["smltok"] = {"i2w": seqdata["smltok"].i2w, "w2i": seqdata["smltok"].w2i,
                             "word_count": dict(seqdata["smltok"].word_count)}

    # dataset["config"] = {"datvocabsize": int, "comvocabsize": int, "smlvocabsize": int, "datlen": int, "comlen": int, "smllen": int}
    dataset["config"] = seqdata["config"]

    print("Save dataset to: %s" % (output_path))
    with open(output_path, 'wb') as output:
        pickle.dump(dataset, output)