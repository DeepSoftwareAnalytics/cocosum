# !/usr/bin/env python
# !-*-coding:utf-8 -*-
import os
import sys
import time
sys.path.append("../../")
from util.Config import Config as cf
from util.LoggerUtil import set_logger, debug_logger
from util.DataUtil import time_format, read_pickle_data, save_pickle_data


def merge(path, data_name):
    d = {}
    # print(os.listdir(path))
    _ = [d.update(read_pickle_data(os.path.join(path, f))) for f in os.listdir(path) if data_name in f]
    # debug_logger(data_name + " done")
    return d


class Config:
    uml_dir = "../../data/csn/uml/"
    m2uid_dir = "../../data/csn/uml/m2uid/"


# ToDO rename .py
if __name__ == '__main__':

    set_logger(cf.DEBUG)
    parts = ['train', 'valid', 'test']
    start_time = time.perf_counter()
    for part in parts:
        m2uid = merge(os.path.join(Config.m2uid_dir, part), "m2uid")
        save_pickle_data(os.path.join(Config.uml_dir, part), 'm2uid.pkl', m2uid)
    debug_logger("time cost %s" % time_format(time.perf_counter() - start_time))
    # m2uid = {part: merge(os.path.join(Config.m2uid_dir, part), "m2uid") for part in parts}
    # umls = {part: merge(os.path.join(Config.uml_dir, part), "umls") for part in parts}
    # big_graph_id = {part: read_pickle_data(os.path.join(Config.uml_dir, part, "big_graph_id.pkl")) for part in parts}

    # save_pickle_data(Config.uml_dir, 'm2uid.pkl', m2uid)
    # save_pickle_data(Config.uml_dir, 'umls.pkl', umls)
    # save_pickle_data(Config.uml_dir, 'big_graph_id.pkl', big_graph_id)
    # debug_logger("time cost %s" % time_format(time.perf_counter() - start_time))


