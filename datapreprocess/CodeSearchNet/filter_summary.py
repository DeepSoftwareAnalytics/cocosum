# !/usr/bin/env python
# !-*-coding:utf-8 -*-
import sys
import time
sys.path.append("../../")
from util.DataUtil import print_time, read_pickle_data, save_pickle_data


def is_ascii(s):
    return all(ord(c) < 128 for c in s)


def is_summary_good(summary):
    if len(summary) < 4 or summary[-1] is "{":
        return False
    for word in summary:
        if not is_ascii(word):
            return False
    return True


def filter_dataset(data):
    correct_summary = {}
    for m_id, method in data.items():
        if is_summary_good(method["docstring_tokens"]):
            correct_summary[m_id] = method
    print("data size", len(data))
    print("count and ratio of filtered summary",  len(correct_summary), len(correct_summary)/len(data))
    return correct_summary
    # write_to_pickle(destination, filter_data, file_name="methods.pkl")
    # write_to_pickle(destination, filter_id, file_name="filter_id")



class ConfigFilterSum:
    root = "../../../Data/csn/"
    # csn_data_path = r"/datadrive/Data/csn/csn_added_id"
    csn_add_id_data_path = r"../../../Data/csn/csn.pkl"
    # filter_data_path = r"/datadrive/Data/csn/csn_after_filter/filtered_sum/"


if __name__ == '__main__':

    dataset_dirs = ['train', 'val', 'test']
    start = time.perf_counter()
    correct_summary_fid = []
    add_id_data = read_pickle_data(ConfigFilterSum.csn_add_id_data_path)
    correct_summary = {}
    correct_summary_fid = {}
    for partition in dataset_dirs:
        correct_summary_partition = filter_dataset(add_id_data[partition])
        correct_summary[partition] = correct_summary_partition
        correct_summary_fid[partition] = list(correct_summary_partition.keys())
        # correct_summary_fid.extend(list(correct_summary_partition.keys()))
        print_time(time.perf_counter() - start)
        start = time.perf_counter()
    save_pickle_data(ConfigFilterSum.root, "correct_summary_lenGT3.pkl", correct_summary)
    save_pickle_data(ConfigFilterSum.root, "correct_summary_fid_lenGT3.pkl", correct_summary_fid)


