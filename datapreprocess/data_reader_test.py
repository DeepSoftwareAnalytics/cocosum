
import numpy as np

from util.DataUtil import read_pickle_data


# def train_test_diff(data):
#     # train_summary_raw = data['ctrain']
#     test_summary_raw = data['ctest']  # dict
#     test_summary = np.array([]).astype(int)
#     for key, seq in test_summary_raw.items():
#         test_summary = np.concatenate((test_summary, seq))
#     test_summary = np.unique(test_summary)
#     print("test_summary_voc_len: %s", len(test_summary))
#
#     train_summary_raw = data['ctrain']
#     train_summary = np.array([]).astype(int)
#     for key, seq in train_summary_raw.items():
#         train_summary = np.concatenate((train_summary, seq))
#     train_summary = np.unique(train_summary)
#     print("train_summary_voc_len: %s", len(train_summary))
#
#     diff = set(train_summary).difference(set(test_summary))
#     diff_words = [data['comstok']['i2w'][i] for i in diff]
#     print(diff)
#     print(diff_words)


def train_test_diff(data):
    train_summary = data["ctrain"]
    train_voc_set = set()

    for key, seq in train_summary.items():
        for token in seq:
            train_voc_set.add(token)

    print("train voc size %d" % (len(train_voc_set)))

    test_summary = data["ctest"]
    test_voc_set = set()

    for key, seq in test_summary.items():
        for token in seq:
            test_voc_set.add(token)

    print("test voc size %d" % (len(test_voc_set)))

    diff_set = test_voc_set.difference(train_voc_set)
    print("disjoint voc (UNK) size %d" % (len(diff_set)))


if __name__ == '__main__':
    # data_path = "../data/data_csn_voc1000_funcomformat_200218/standard_data.pkl"
    # data_path = "../data/funcom_icse19_remove_dependency_small_100/standard_data.pkl"
    data_path = "../data/funcom_icse19_remove_dependency/standard_data.pkl"

    data = read_pickle_data(data_path)
    # print(summary_n_unk(data))
    train_test_diff(data)