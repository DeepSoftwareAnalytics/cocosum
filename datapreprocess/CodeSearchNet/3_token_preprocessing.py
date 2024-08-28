# refer to: http://13.76.197.150/notebooks/mnt/Data/csn_added_id/split_and_filter_stopwords.ipynb
import os
import pickle
import sys
import time
sys.path.append("../../")
from datapreprocess.CodeSearchNet.get_code_tokens import tokenize_source_code
from datapreprocess.CodeSearchNet.get_sbt_csn import write_code_to_java_file, get_sbt_str
from util.DataUtil import save_pickle_data,  time_format, code_tokens_replace_str_num, str_to_bool, \
    filter_punctuation_pl, code_tokens_split_identifier, lower_case_str_arr
from util.Config import Config as cf
from util.LoggerUtil import set_logger, debug_logger
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
        root = cf.processed_data_path
        # input :   root + "csn.pkl"
        # output: 
                1. java file in  root +"java_files"
                2. SBT in  root +  "sbt/sbts.pkl"
                3. tokenized SBT in  root + "sbt/sbts_tokens.pkl"
        """
        debug_logger("Generate SBT")
        start_time = time.perf_counter()
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
        debug_logger("time cost %s" % time_format(time.perf_counter() - start_time))
    # process code.
    code_dict = get_code_dict(data)
    debug_logger("Process code")
    start_time = time.perf_counter()
    for partition in code_dict:
        if is_djl:
            code_dict[partition] = update_dict(code_dict[partition], tokenize_source_code)
        if is_dr:
            code_dict[partition] = update_dict(code_dict[partition], code_tokens_replace_str_num)
        if is_dsi:
            code_dict[partition] = update_dict(code_dict[partition], code_tokens_split_identifier)
        if is_dfp:
            code_dict[partition] = update_dict(code_dict[partition], filter_punctuation_pl)
        if is_dlc:
            code_dict[partition] = update_dict(code_dict[partition], lower_case_str_arr)
    save_pickle_data(os.path.join(cf.processed_data_path, 'code'), code_file_name, code_dict)
    debug_logger("time cost %s" % time_format(time.perf_counter() - start_time))

    # process summary
    summary_dict = get_summary_dict(data)
    debug_logger("Process summary")
    start_time = time.perf_counter()
    for partition in summary_dict.keys():
        if is_cfp:
            summary_dict[partition] = update_dict(summary_dict[partition], filter_punctuation_pl)
        if is_csi:
            summary_dict[partition] = update_dict(summary_dict[partition], code_tokens_split_identifier)
        if is_clc:
            summary_dict[partition] = update_dict(summary_dict[partition], lower_case_str_arr)
    save_pickle_data(os.path.join(cf.processed_data_path, 'summary'), summary_file_name, summary_dict)
    debug_logger("time cost %s" % time_format(time.perf_counter() - start_time))


if __name__ == '__main__':
    # TODO: skip this step if **.pkl already exits.
    st = time.perf_counter()
    set_logger(cf.DEBUG)
    process(is_cfp=True, is_csi=True, is_clc=True, is_djl=True, is_dsi=True, is_dfp=False, is_dlc=True, is_dr=True, is_gen_sbt=True)
    debug_logger("time cost %s" % time_format(time.perf_counter() - st))
