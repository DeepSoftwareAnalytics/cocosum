# refer to: http://13.76.197.150/notebooks/mnt/Data/csn_added_id/split_and_filter_stopwords.ipynb
import os
import sys
import time

sys.path.append("../../")
from datapreprocess.CodeSearchNet.build_csn_vocab import count_word_parallel, build_vocab_info
from util.DataUtil import time_format, read_pickle_data, get_all_tokens, save_pickle_data, save_json_data
from util.Config import Config as cf
from util.LoggerUtil import set_logger, debug_logger


def word_count(token_path):
    """
    Take a pickle dataset path, return a Counter on train set.
    """
    tokens = read_pickle_data(token_path)
    word_list = get_all_tokens(tokens["train"])
    token_word_count = count_word_parallel(word_list)
    return token_word_count


class Config:
    # dataset_path = '../../data/csn/'
    data_path = '../../data/csn/'
    package_wise = False
    method_wise = False
    code_tokens_file_name = "djl1_dfp0_dsi1_dlc1_dr1.pkl"
    summary_tokens_file_name = "cfp1_csi1_cfd0_clc1.pkl"
    sbt_tokens_file_name = 'sbts_tokens.pkl'
    voc_info_file_name = 'csn_trainingset_' + code_tokens_file_name.split(".")[0] + "_" + \
                         summary_tokens_file_name.split(".")[0] + '.json'


def build_voc(cwc=False, dwc=False, swc=False):
    """
    generate vocab_raw/code_word_count_djl1_dfp0_dsi1_dlc0_dr1.pkl
    vocab_raw/csn_trainingset_djl1_dfp0_dsi1_dlc1_dr1_cfp1_csi1_cfd0_clc1.json
    vocab_raw/sbt_word_count.pkl
    vocab_raw/summary_word_count_cfp1_csi1_cfd0_clc1.pkl
    """

    if Config.package_wise:
        Config.data_path = os.path.join(Config.data_path, "package_wise")
    if Config.method_wise:
        Config.data_path = os.path.join(Config.data_path, "method_wise")
    path = os.path.join(Config.data_path, "vocab_raw")
    if cwc:
        summary_tokens_path = os.path.join(Config.data_path, "summary", Config.summary_tokens_file_name)
        summary_word_count = word_count(summary_tokens_path)
        save_pickle_data(path, "summary_word_count_" + Config.summary_tokens_file_name, summary_word_count)
    if dwc:
        code_tokens_path = os.path.join(Config.data_path, "code", Config.code_tokens_file_name)
        code_word_count = word_count(code_tokens_path)
        save_pickle_data(path, "code_word_count_" + Config.code_tokens_file_name, code_word_count)
    if swc:
        sbt_tokens_path = os.path.join(Config.data_path, "sbt", Config.sbt_tokens_file_name)
        sbt_word_count = word_count(sbt_tokens_path)
        save_pickle_data(path, "sbt_word_count.pkl", sbt_word_count)
    vocab = build_vocab_info(code_word_count, summary_word_count, sbt_word_count)
    save_json_data(path, Config.voc_info_file_name, vocab)


if __name__ == '__main__':
    start_time = time.perf_counter()
    set_logger(cf.DEBUG)
    build_voc(cwc=True, dwc=True, swc=True)
    debug_logger("time cost %s" % time_format(time.perf_counter() - start_time))
