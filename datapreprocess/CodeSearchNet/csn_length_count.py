# -*- coding: utf-8 -*-
# @Time      : 2020-02-21 15:35
# @Author    : Eason Hu
# @Site      : 
# @File      : csn_lengths_count.py
import sys
from util.DataUtil import read_pickle_data, padding, split_and_filter_identifier, count_word, \
    build_vocab_with_pad_unk_sos_eos, extract_col, list_flatten, save_pickle_data, array_split, \
    save_json_data, split_and_filter_identifier_lists_parallel, split_and_filter_identifier_lists

from collections import Counter
import os
import gzip
import json
from util.Config import Config as config
import pickle
import time
from spiral import ronin
import string
import matplotlib.pyplot as plt

sys.path.append("../../")


# 1. yuxuan TODO, delete, don't reinvent
# 2. don't fuse input with output, token_count_len
def count_len(token_count_len, tokens):
    if len(tokens) in token_count_len:
        token_count_len[len(tokens)] += 1
    else:
        token_count_len[len(tokens)] = 1
    return token_count_len


# yuxuan TODO, review suggestions:
# 1. please don't write duplicate code! 80%, 90% should be a parameter.
# 2. don't tangle the 80% 90% quantile logic with the plot logic
# 3. be careful about variable names: e.g. L?
def parse_result(lengths_dict):
    L = sorted(lengths_dict.items(), key=lambda item: int(item[0]))
    all_len = sum(lengths_dict.values())
    cur_len = 0
    res = 0
    xs = []
    ys = []
    print('max_len:', L[-1][0])
    print('min_len:', L[0][0])
    for i in range(len(lengths_dict)):
        cur_len += L[i][1]
        if cur_len >= all_len * 0.9:
            print('90%:', L[i][0])
            break
        xs.append(int(L[i][0]))
        ys.append(L[i][1])

    plt.plot(xs, ys)
    plt.xlabel('lengths')
    plt.ylabel('counts')
    plt.show()
    cur_len = 0
    for i in range(len(lengths_dict)):
        cur_len += L[i][1]
        if cur_len >= all_len * 0.8:
            res = L[i][0]
            print('80%:', res)
            break
    return res


def get_length_count(data, col):
    docstring_list = extract_col(data, col)
    docstring_split_list = split_and_filter_identifier_lists_parallel(docstring_list)
    docstring_lengths = [len(d) for d in docstring_split_list]
    length_count = Counter(docstring_lengths)
    return length_count


if __name__ == '__main__':
    # load data
    start = time.perf_counter()
    train_data = read_pickle_data('../../../Data/csn/csn_extracted_append_id/train/data.pkl')
    print("read pickle data time train: ", time.perf_counter() - start)

    print('code length info: ')
    start = time.perf_counter()
    parse_result(get_length_count(train_data, 'docstring_tokens'))
    print("process docstring_tokens time: ", time.perf_counter() - start)

    print('\nSummary length info: ')
    start = time.perf_counter()
    parse_result(get_length_count(train_data, 'code_tokens'))
    print("process code_tokens time: ", time.perf_counter() - start)
