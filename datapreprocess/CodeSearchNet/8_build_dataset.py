# refer to: http://13.76.197.150/notebooks/mnt/Data/csn_added_id/split_and_filter_stopwords.ipynb
import os
import time
import sys
import argparse

sys.path.append("../../")
from util.DataUtil import save_pickle_data, get_file_name, str_to_bool, time_format, padding, \
    read_pickle_data, load_vocab
from util.Config import Config as cf
from util.LoggerUtil import set_logger, info_logger, debug_logger


def code2ids(tokens, w2i, seq_len):
    unk_id = w2i[cf.UNK_token]
    ids = [w2i.get(token, unk_id) for token in tokens[:seq_len]]
    ids = padding(ids, seq_len, cf.PAD_token_id)
    return ids


def summary2ids(summary_tokens, summary_w2i):
    summary_unk_id = summary_w2i[cf.UNK_token]
    summary_ids = [summary_w2i.get(token, summary_unk_id) for token in summary_tokens[:cf.summary_len - 1]]
    summary_ids.insert(0, cf.SOS_token_id)
    if len(summary_ids) < cf.summary_len:
        summary_ids.append(cf.EOS_token_id)
    summary_ids = padding(summary_ids, cf.summary_len, cf.PAD_token_id)
    return summary_ids


def process_data(summary, code, sbt, code_w2i, summary_w2i, sbt_w2i, fids):
    code_ids = {}
    summary_ids = {}
    sbt_ids = {}

    for fid in fids:
        code_ids[fid] = code2ids(code[fid], code_w2i, cf.code_len)
        summary_ids[fid] = summary2ids(summary[fid], summary_w2i)
        sbt_ids[fid] = code2ids(sbt[fid], sbt_w2i, cf.sbt_len)
    return code_ids, summary_ids, sbt_ids


def get_dataset(vocab_size, enable_uml=True):
    code_word_count, summary_word_count, sbt_word_count, \
    summary_w2i, summary_i2w, code_i2w, code_w2i, \
    sbt_i2w, sbt_w2i = load_vocab(os.path.join(cf.processed_data_path, "vocab_raw", cf.voc_info_file_name),
                                  vocab_size)

    summary_tokens = read_pickle_data(
        os.path.join(cf.processed_data_path, "summary", cf.summary_tokens_file_name))
    code_tokens = read_pickle_data(os.path.join(cf.processed_data_path, "code", cf.code_tokens_file_name))
    sbt_tokens = read_pickle_data(os.path.join(cf.processed_data_path, "sbt", cf.sbt_tokens_file_name))

    correct_id_filename = "cfd{}_sfd1_ufd{}_fid.pkl".format(str(cf.summary_filter_bad_cases + 0),
                                                            str(cf.uml_filter_data_bad_cases + 0))
    correct_fid_path = os.path.join(cf.processed_data_path, "correct_fid", correct_id_filename)
    correct_fid = read_pickle_data(correct_fid_path)

    dtrain, ctrain, strain = process_data(summary_tokens["train"], code_tokens["train"],
                                          sbt_tokens["train"], code_w2i, summary_w2i, sbt_w2i, correct_fid["train"])
    dval, cval, sval = process_data(summary_tokens["valid"], code_tokens["valid"],
                                    sbt_tokens["valid"], code_w2i, summary_w2i, sbt_w2i, correct_fid["valid"])
    dtest, ctest, stest = process_data(summary_tokens["test"], code_tokens["test"],
                                       sbt_tokens["test"], code_w2i, summary_w2i, sbt_w2i, correct_fid["test"])

    dataset = {"ctrain": ctrain, "cval": cval, "ctest": ctest,
               "dtrain": dtrain, "dval": dval, "dtest": dtest,
               "strain": strain, "sval": sval, "stest": stest,
               "comstok": {"i2w": summary_i2w, "w2i": summary_w2i, "word_count": summary_word_count},
               "datstok": {"i2w": code_i2w, "w2i": code_w2i, "word_count": code_word_count},
               "smlstok": {"i2w": sbt_i2w, "w2i": sbt_w2i, "word_count": sbt_word_count},
               "config": {"datvocabsize": len(code_i2w), "comvocabsize": len(summary_i2w),
                          "smlvocabsize": len(sbt_i2w), "datlen": cf.code_len,
                          "comlen": cf.summary_len, "smllen": cf.sbt_len}}

    if enable_uml:
        m2u_m2c = read_pickle_data(cf.m2u_m2c_path)
        dataset.update(m2u_m2c)

    return dataset


def build_dataset():
    vocab = {"code": cf.code_vocab_size, "summary":
        cf.summary_vocab_size, "sbt": cf.sbt_vocab_size}
    dataset = get_dataset(vocab, False)
    filename = get_file_name()
    save_pickle_data(cf.dataset_path, filename, dataset)


def set_config():
    parser = argparse.ArgumentParser()
    # code processing  / 5
    parser.add_argument('-djl', "--code_tokens_javalang_results", type=str, choices=["True", "False"], required=False)
    parser.add_argument('-dfp', "--code_filter_punctuation", type=str, choices=["True", "False"], required=False)
    parser.add_argument('-dsi', "--code_split_identifier", type=str, choices=["True", "False"], required=False)
    parser.add_argument('-dlc', "--code_lower_case", type=str, choices=["True", "False"], required=False)
    parser.add_argument('-dr', "--code_replace_string_num", type=str, choices=["True", "False"], required=False)
    # summary processing/3
    parser.add_argument('-cfp', "--summary_filter_punctuation", type=str, choices=["True", "False"], required=False)
    parser.add_argument('-csi', "--summary_split_identifier", type=str, choices=["True", "False"], required=False)
    parser.add_argument('-cfd', "--summary_filter_bad_cases", type=str, choices=["True", "False"], required=False)
    # seq len /3
    parser.add_argument('-dlen', "--code_len", required=False)
    parser.add_argument('-clen', "--summary_len", required=False)
    parser.add_argument('-slen', "--sbt_len", required=False)
    # voc size /3
    parser.add_argument('-dvoc', "--code_vocab_size", required=False)
    parser.add_argument('-cvoc', "--summary_vocab_size", required=False)
    parser.add_argument('-svoc', "--sbt_vocab_size", required=False)
    # dataset
    parser.add_argument('-dataset_path', type=str, required=False)
    #  is package wise
    parser.add_argument('-pkg', "--package_wise", type=str, choices=["True", "False"], required=False)
    #  is method wise
    parser.add_argument('-mtd', "--method_wise", type=str, choices=["True", "False"], required=False)
    args = parser.parse_args()

    # code processing
    if args.code_tokens_javalang_results:
        cf.code_tokens_using_javalang_results = str_to_bool(args.code_tokens_javalang_results)
    if args.code_filter_punctuation:
        cf.code_filter_punctuation = str_to_bool(args.code_filter_punctuation)
    if args.code_split_identifier:
        cf.code_split_identifier = str_to_bool(args.code_split_identifier)
    if args.code_lower_case:
        cf.code_lower_case = str_to_bool(args.code_lower_case)
    if args.code_replace_string_num:
        cf.code_replace_string_num = str_to_bool(args.code_replace_string_num)

    # summary processing
    if args.summary_filter_punctuation:
        cf.summary_filter_punctuation = str_to_bool(args.summary_filter_punctuation)
    if args.summary_split_identifier:
        cf.summary_split_identifier = str_to_bool(args.summary_split_identifier)
    if args.summary_filter_bad_cases:
        cf.summary_filter_bad_cases = str_to_bool(args.summary_filter_bad_cases)

    # seq len
    if args.code_len:
        cf.code_len = int(args.code_len)
    if args.summary_len:
        cf.summary_len = int(args.summary_len)
    if args.sbt_len:
        cf.sbt_len = int(args.sbt_len)

    # voc size
    if args.code_vocab_size:
        cf.code_vocab_size = int(args.code_vocab_size)
    if args.summary_vocab_size:
        cf.summary_vocab_size = int(args.summary_vocab_size)
    if args.sbt_vocab_size:
        cf.sbt_vocab_size = int(args.sbt_vocab_size)

    # dataset
    if args.dataset_path:
        cf.dataset_path = args.dataset_path

    #  package_wise
    if args.package_wise:
        cf.package_wise = str_to_bool(args.package_wise)

    #  method_wise
    if args.method_wise:
        cf.method_wise = str_to_bool(args.method_wise)
    if cf.package_wise:
        cf.processed_data_path = os.path.join(cf.processed_data_path, "package_wise")
    if cf.method_wise:
        cf.processed_data_path = os.path.join(cf.processed_data_path, "method_wise")


def basic_info_logger():
    info_logger("[Setting] DEBUG: %s" % (str(cf.DEBUG)))
    info_logger("[Setting] code_tokens_using_javalang_results: %s" % (str(cf.code_tokens_using_javalang_results)))
    info_logger("[Setting] code_filter_punctuation : %s" % (str(cf.code_filter_punctuation)))
    info_logger("[Setting] code_split_identifier: %s" % (str(cf.code_split_identifier)))
    info_logger("[Setting] code_lower_case: %s" % (str(cf.code_lower_case)))
    info_logger("[Setting] code_replace_string_num: %s" % (str(cf.code_replace_string_num)))

    info_logger("[Setting] summary_filter_punctuation: %s" % (str(cf.summary_filter_punctuation)))
    info_logger("[Setting] summary_split_identifier: %s" % (str(cf.summary_split_identifier)))
    info_logger("[Setting] summary_filter_bad_cases: %s" % (str(cf.summary_filter_bad_cases)))

    info_logger("[Setting] code_len: %d" % cf.code_len)
    info_logger("[Setting] summary_len: %d" % cf.summary_len)
    info_logger("[Setting] sbt_len: %d" % cf.sbt_len)

    info_logger("[Setting] code_vocab_size: %d" % cf.code_vocab_size)
    info_logger("[Setting] summary_vocab_size: %d" % cf.summary_vocab_size)
    info_logger("[Setting] sbt_vocab_size: %d" % cf.sbt_vocab_size)

    info_logger("[Setting]  uml_filter_data_bad_cases: %s" % cf.uml_filter_data_bad_cases)

    info_logger("[Setting]  save file in : %s" % cf.dataset_path)


if __name__ == '__main__':
    start_time = time.perf_counter()
    set_logger(cf.DEBUG)
    set_config()
    basic_info_logger()
    build_dataset()
    debug_logger("time cost %s" % time_format(time.perf_counter() - start_time))
