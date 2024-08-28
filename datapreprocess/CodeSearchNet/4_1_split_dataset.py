# !/usr/bin/env python
# !-*-coding:utf-8 -*-
import os
import pickle
import sys
import time
sys.path.append("../../")
from util.Config import Config as cf
from util.LoggerUtil import set_logger, debug_logger
from util.DataUtil import make_directory, time_format, save_pickle_data


def dict_slice(adict, start, end):
    return {k: adict[k] for k in list(adict.keys())[start:end]}


def split_dataset(data, splitted_data_dir):
    data_size, length = Config.small_data_size, len(data)
    for start in range(0, length, Config.small_data_size):
        end = start + Config.small_data_size
        file_name = str(start) + "_" + str(min(end, length)) + '.pkl'
        small_data = dict_slice(data, start, end)
        save_pickle_data(splitted_data_dir, file_name, small_data)


class Config:
    large_data_path = "../../data/csn/csn.pkl"
    small_data_path = "../../data/csn/split_data/"
    small_data_size = 3000


if __name__ == '__main__':
    set_logger(cf.DEBUG)
    debug_logger("Each small data size: %d " % Config.small_data_size)
    parts = ['train', 'valid', 'test']
    start_time = time.perf_counter()
    csn = pickle.load(open(Config.large_data_path, "rb"))
    for part in parts:
        splitted_data_path = os.path.join(Config.small_data_path, part)
        make_directory(splitted_data_path)
        split_dataset(csn[part], splitted_data_path)
        debug_logger("Time cost %s" % time_format(time.perf_counter() - start_time))
        start_time = time.perf_counter()
