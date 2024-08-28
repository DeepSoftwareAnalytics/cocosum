# refer to: http://13.76.197.150/notebooks/mnt/Data/csn_added_id/split_and_filter_stopwords.ipynb
import os
import pickle
import sys
import time
import argparse
sys.path.append("../../")
from datapreprocess.CodeSearchNet.get_code_tokens import tokenize_source_code
from datapreprocess.CodeSearchNet.get_sbt_csn import write_code_to_java_file, get_sbt_str
from datapreprocess.CodeSearchNet.build_dataset import get_dataset
from datapreprocess.CodeSearchNet.build_csn_vocab import count_word_parallel, build_vocab_info
from util.DataUtil import print_time, read_pickle_data, get_all_tokens, save_pickle_data, save_json_data, \
     get_file_name, code_tokens_replace_str_num, \
    filter_punctuation_pl, code_tokens_split_identifier, lower_case_str_arr
from util.Config import Config as cf
from util.LoggerUtil import set_logger,info_logger
from multiprocessing import cpu_count, Pool


def extract_cols(data, ids, cols):
    res = []
    for i in ids:
        res.append([data[i][col] for col in cols])
    return res


def add_key(d):
    """
    Take a dictionary d {k:v}, return {k:(k,v)}
    """
    res = {}
    for k, v in d.items():
        res[k] = (k, v)
    return res


def update_dict(d, f):
    """
    Update dict d = {k:v} with f, return {k:f(v)}
    :param d:
    :param f:
    :return:
    """
    cores = cpu_count()
    pool = Pool(cores)
    keys = []
    values = []
    for i in list(d):
        keys.append(i)
        values.append(d[i])
    results = pool.map(f, values)
    pool.close()
    pool.join()
    new_d = {}
    for k, v in zip(keys, results):
        new_d[k] = v
    return new_d


def split_by_whitespace(s):
    return s.split(" ")


def select_dict_value(data, col):
    res_dict = {}
    for k, v in data.items():
        res_dict[k] = v[col]
    return res_dict


def get_code_dict(data):
    """
    Generate code_dict = {k:{fid:code}}, k in ['train', 'val', 'test'].
    :param data:
    :return:
    """
    code_dict = {}
    for partition in data.keys():
        code_dict[partition] = select_dict_value(data[partition], 'code')
        for k, v in code_dict[partition].items():
            code_dict[partition][k] = v + '\n'
    return code_dict


def get_summary_dict(data):
    """
    Generate summary_dict = {k:{fid:[docstring_tokens]}}, k in ['train', 'val', 'test'].
    :param data:
    :return:
    """
    summary_dict = {}
    for partition in data.keys():
        summary_dict[partition] = select_dict_value(data[partition], 'docstring_tokens')
    return summary_dict


def process(is_csi=False, is_clc=False, is_cfp=False,
            is_djl=False, is_dfp=False, is_dsi=False, is_dlc=False, is_dr=False,
            is_gen_sbt=False):
    """
     This function will generate:
        code/djl1_dfp0_dsi1_dlc1_dr1.pkl
        summary/cfp1_csi1_cfd0_clc1.pkl
        sbt/sbts.pkl
        sbt/sbts_tokens.pkl
    """
    code_file_name = 'djl{}_dfp{}_dsi{}_dlc{}_dr{}.pkl'.format(str(is_djl + 0), str(is_dfp + 0), str(is_dsi + 0),
                                                               str(is_dlc + 0), str(is_dr + 0))
    summary_file_name = 'cfp{}_csi{}_cfd0_clc{}.pkl'.format(str(is_cfp + 0), str(is_csi + 0), str(is_clc + 0))

    data = pickle.load(open(os.path.join(cf.processed_data_path, 'csn.pkl'), "rb"))

    # process sbt.
    if is_gen_sbt:
        """
        Generate SBT given csn.pkl, which just add fid in original dataset
        # input : "../../../Data/csn/csn.pkl"
        # output: 
                1. java file in "../../../Data/java_files"
                2. SBT in "../../../Data/csn/sbts.pkl"
                3. tokenized SBT in "../../../Data/csn/sbts_tokens.pkl"
        """
        code_dict = get_code_dict(data)
        code_dict_with_key = {}
        sbt_dict = {}
        sbt_tokens_dict = {}
        for partition in code_dict:
            code_dict_with_key[partition] = add_key(code_dict[partition])
            update_dict(code_dict_with_key[partition], write_code_to_java_file)
            sbt_dict[partition] = update_dict(code_dict_with_key[partition], get_sbt_str)
            sbt_tokens_dict[partition] = update_dict(sbt_dict[partition], split_by_whitespace)
        # save_pickle_data(config.processed_data_path, 'sbt/sbts.pkl', sbt_dict)
        save_pickle_data(os.path.join(cf.processed_data_path, "sbt"), 'sbts.pkl', sbt_dict)
        # save_pickle_data(config.processed_data_path, 'sbt/sbts_tokens.pkl', sbt_tokens_dict)
        save_pickle_data(os.path.join(cf.processed_data_path, "sbt"), 'sbts_tokens.pkl', sbt_tokens_dict)

    # process code.
    code_dict = get_code_dict(data)
    for partition in code_dict:
        if is_djl:
            code_dict[partition] = update_dict(code_dict[partition], tokenize_source_code)
        if is_dsi:
            code_dict[partition] = update_dict(code_dict[partition], code_tokens_split_identifier)
        if is_dr:
            code_dict[partition] = update_dict(code_dict[partition], code_tokens_replace_str_num)
        if is_dfp:
            code_dict[partition] = update_dict(code_dict[partition], filter_punctuation_pl)
        if is_dlc:
            code_dict[partition] = update_dict(code_dict[partition], lower_case_str_arr)
    save_pickle_data(os.path.join(cf.processed_data_path, 'code'), code_file_name, code_dict)

    # process summary
    summary_dict = get_summary_dict(data)
    for partition in summary_dict.keys():
        if is_clc:
            summary_dict[partition] = update_dict(summary_dict[partition], lower_case_str_arr)
        if is_cfp:
            summary_dict[partition] = update_dict(summary_dict[partition], filter_punctuation_pl)
        if is_csi:
            summary_dict[partition] = update_dict(summary_dict[partition], code_tokens_split_identifier)
    save_pickle_data(os.path.join(cf.processed_data_path, 'summary'), summary_file_name, summary_dict)


def word_count(token_path):
    """
    Take a pickle dataset path, return a Counter on train set.
    """
    tokens = read_pickle_data(token_path)
    word_list = get_all_tokens(tokens["train"])
    token_word_count = count_word_parallel(word_list)
    return token_word_count


def build_voc(cwc=False, dwc=False, swc=False):
    """
    generate vocab_raw/code_word_count_djl1_dfp0_dsi1_dlc0_dr1.pkl
    vocab_raw/csn_trainingset_djl1_dfp0_dsi1_dlc1_dr1_cfp1_csi1_cfd0_clc1.json
    vocab_raw/sbt_word_count.pkl
    vocab_raw/summary_word_count_cfp1_csi1_cfd0_clc1.pkl
    """
    if cf.package_wise:
        cf.processed_data_path = os.path.join(cf.processed_data_path, "package_wise")
    if cf.method_wise:
        cf.processed_data_path = os.path.join(cf.processed_data_path, "method_wise")
    path = os.path.join(cf.processed_data_path, "vocab_raw")
    if cwc:
        summary_tokens_path = os.path.join(cf.processed_data_path, "summary", cf.summary_tokens_file_name)
        summary_word_count = word_count(summary_tokens_path)
        save_pickle_data(path, "summary_word_count_" + cf.summary_tokens_file_name, summary_word_count)
    if dwc:
        code_tokens_path = os.path.join(cf.processed_data_path, "code", cf.code_tokens_file_name)
        code_word_count = word_count(code_tokens_path)
        save_pickle_data(path, "code_word_count_" + cf.code_tokens_file_name, code_word_count)
    if swc:
        sbt_tokens_path = os.path.join(cf.processed_data_path, "sbt", cf.sbt_tokens_file_name)
        sbt_word_count = word_count(sbt_tokens_path)
        save_pickle_data(path, "sbt_word_count.pkl", sbt_word_count)
    vocab = build_vocab_info(code_word_count, summary_word_count, sbt_word_count)
    save_json_data(path, cf.voc_info_file_name, vocab)


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

    info_logger("[Setting]  uml_filter_data_bad_cases: %d" % cf.uml_filter_data_bad_cases)

    info_logger("[Setting]  save file in : %s" % cf.dataset_path)


if __name__ == '__main__':
    start_time = time.perf_counter()
    set_logger(cf.DEBUG)
    set_config()
    basic_info_logger()
    process(is_cfp=True, is_csi=True, is_djl=True, is_dsi=True, is_dlc=False, is_dr=True, is_gen_sbt=True)
    build_voc(cwc=True, dwc=True, swc=True)
    build_dataset()
    print(time.perf_counter() - start_time)
