# !/usr/bin/env python
# !-*-coding:utf-8 -*-
import os
import pickle
import sys
import time
import traceback

from tqdm import tqdm

sys.path.append("../../")
from util.Config import Config as cf
from util.LoggerUtil import set_logger, debug_logger
from util.DataUtil import time_format, read_pickle_data, save_pickle_data


def filter_class_not_in_uml(data_part, methods):
    # umls = pickle.load(open(os.path.join(Config.uml_dir, data_part, "umls.pkl"), "rb"))
    # methods = pickle.load(open(os.path.join(Config.uml_dir, data_part, "methods.pkl"), "rb"))
    # method2uml = pickle.load(open(os.path.join(Config.uml_dir, data_part, "method2uml_index.pkl"), "rb"))

    umls = read_pickle_data(os.path.join(Config.uml_dir, data_part, "umls.pkl"))
    # methods = read_pickle_data(os.path.join(Config.uml_dir, data_part, "methods.pkl"))
    method2uml = read_pickle_data(os.path.join(Config.uml_dir, data_part, "m2uid.pkl"))

    correct_method = {}
    correct_fid = []
    # try:
    for fid in tqdm(method2uml.keys(), desc=data_part):
        try:
            method = methods[fid]
            class_name_of_method = method["func_name"].split(".")[0]
            uml = umls[method2uml[fid]]
            wrong_uml = False
            for edge in uml['edge_information']:
                if not(edge[0] in uml['nodes'] and edge[1] in uml['nodes']):
                    wrong_uml = True
                    break
            if wrong_uml:
                continue
            for _, node in uml['nodes_information'].items():
                class_name_in_package = node["class_declaration"]["name"]
                if class_name_of_method in class_name_in_package.split("<"):
                # if class_name_of_method in class_name_in_package:
                    correct_method[fid] = method
                    correct_fid.append(fid)
                    break
        except KeyError:
            error_file = os.path.join(Config.uml_dir, part + "_filter_error.txt")
            traceback.print_exc(file=open(error_file, 'a'))

    # try:
    debug_logger("length of methods: %d " % len(methods.keys()))
    debug_logger("length of method_class_in_package: %d " % len(correct_fid))
    # except:
    #     pass
    # return correct_fid, umls, method2uml
    return correct_fid, correct_method


class Config:
    # file_path = r"/mnt/enshi/uml/all_data_uml/"
    uml_dir = "../../data/csn/uml/"
    csn_path = "../../data/csn/csn.pkl"


if __name__ == '__main__':
    set_logger(cf.DEBUG)
    parts = ["train", 'valid', 'test']
    correct_uml_fid = {}
    csn = pickle.load(open(Config.csn_path, "rb"))
    start_time = time.perf_counter()
    for part in parts:
        debug_logger(part)
        partition_correct_uml_fid, correct_methods = filter_class_not_in_uml(part, csn[part])
        correct_uml_fid[part] = partition_correct_uml_fid
        save_pickle_data(os.path.join(Config.uml_dir, part), 'methods.pkl', correct_methods)
    save_pickle_data(Config.uml_dir, 'correct_uml_fid.pkl', correct_uml_fid)
    debug_logger("time cost %s" % time_format(time.perf_counter() - start_time))
