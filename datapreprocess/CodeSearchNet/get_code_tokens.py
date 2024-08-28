# !/usr/bin/env python
# !-*-coding:utf-8 -*-
import os
import pickle
import sys
import time
from multiprocessing import cpu_count, Pool

import javalang

sys.path.append("../../")
from util.DataUtil import make_directory, print_time, array_split, save_pickle_data


def tokenize_source_code_list(codes):
    return [tokenize_source_code(code) for code in codes]


def tokenize_source_code(code_string):
    """
    Generate a list of string after javalang tokenization.
    :param code_string: a string of source code
    :return:
    """
    code_string.replace("#", "//")
    tokens = list(javalang.tokenizer.tokenize(code_string))
    return [token.value for token in tokens]


def get_code_tokens(data):
    cores = cpu_count()
    pool = Pool(cores)
    ids = list(data.keys())
    ids_split = array_split(ids, cores)
    data_split = []

    for split in ids_split:
        data_split.append([data[i]["code"] + '\n' for i in split])

    results = pool.map(tokenize_source_code_list, data_split)
    codes = {}
    for ids, result in zip(ids_split, results):
        for i in range(len(ids)):
            mid = ids[i]
            codes[mid] = result[i]
    pool.close()
    pool.join()
    return codes



# def get_code_tokens(data):
#     codes = {}
#     for m_id, method in data.items():
#         codes[m_id] = tokenize_source_code(method["code"] + '\n')
#
#     return codes


def get_save_code_tokens(source, destination):

    codes = {}

    for r, d, f in os.walk(source):
        for file_name in f:
            if file_name.endswith("dataset_with_id.pkl"):
                print(os.path.join(r, file_name))
                data = pickle.load(open(os.path.join(r, file_name), "rb"))
                for m_id, method in data.items():
                    # code = {"code": method["code"]}
                    # code["code_tokens"] =
                    codes[m_id] = tokenize_source_code(method["code"] + '\n')

                print("data size", len(data))
                print("source code count", len(codes), len(codes) / len(data))

    save_pickle_data(destination, "code_tokens.pkl", codes)


class Config:

    csn_data_path = r"/mnt/Data/csn/csn_added_id/"
    code_tokens_path = r"/mnt/Data/csn/code_token_javalang/"

    # csn_data_path = r"/datadrive/Data/csn/csn_added_id/"
    # code_tokens_path = r"/datadrive/Data/csn/code_token_javalang/"

    # for test
    # csn_data_path = r"D:\MSRA\yanlcodesum\1k\filter"
    # code_tokens_path = r"D:\MSRA\yanlcodesum\1k\filter"


if __name__ == '__main__':

    # parser = argparse.ArgumentParser()
    # parser.add_argument("-p", dest='data_dir', choices=['train', 'valid', 'test'], type=str, default='train')
    # args = parser.parse_args()
    # dataset_dir = args.data_dir
    start_time = time.perf_counter()
    dataset_dirs = ['train', 'valid', 'test']
    # dataset_dirs = ['valid']
    for dataset_dir in dataset_dirs:
        csn_data_path = os.path.join(Config.csn_data_path, dataset_dir)
        code_tokens_path = os.path.join(Config.code_tokens_path, dataset_dir)

        make_directory(code_tokens_path)
        get_save_code_tokens(csn_data_path, code_tokens_path)
        print_time(time.perf_counter() - start_time)
