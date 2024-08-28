# !/usr/bin/env python
# !-*-coding:utf-8 -*-
import os
import pickle
import sys
import time

sys.path.append("../../")
from util.DataUtil import print_time, save_pickle_data


def filter_class_not_in_uml(dir_type):

    umls = pickle.load(open(os.path.join(ConfigFilter.file_path, dir_type, "umls.pkl"), "rb"))
    methods = pickle.load(open(os.path.join(ConfigFilter.file_path, dir_type, "methods.pkl"), "rb"))
    method2uml = pickle.load(open(os.path.join(ConfigFilter.file_path, dir_type, "method2uml_index.pkl"), "rb"))

    # method_class_in_package = {}
    method_class_in_package = []
    for idx, method in methods.items():
        class_name_of_method = method["func_name"].split(".")[0]
        uml = umls[method2uml[idx]]
        # uml['nodes_info'] = uml['nodes_information']
        for _, node in uml['nodes_information'].items():
            class_name_in_package = node["class_declaration"]["name"]
            if class_name_of_method in class_name_in_package.split("<"):
                # method_class_in_package[idx] = method
                method_class_in_package.append(idx)
                break
    print("length of methods: ", len(methods.keys()))
    print("length of method_class_in_package: ", len(method_class_in_package))
    return method_class_in_package, umls, method2uml
    # write_to_pickle(os.path.join(Config.file_path, dir_type), method_class_in_package,
    #                 file_name='method_class_in_package.pkl', )


class ConfigFilter:
    file_path = r"/mnt/enshi/uml/all_data_uml/"


if __name__ == '__main__':

    dataset_dirs = ["train", 'val', 'test']
    correct_uml_fid = {}
    umls_all = {}
    method2uml_all = {}
    start_time = time.perf_counter()
    for partition in dataset_dirs:
        print(partition)
        partition_correct_uml_fid, partition_umls, partition_method2uml = filter_class_not_in_uml(partition)
        correct_uml_fid[partition] = partition_correct_uml_fid
        umls_all [partition] = partition_umls
        method2uml_all[partition] = partition_method2uml
        print_time(time.perf_counter() - start_time)
        start_time = time.perf_counter()
    save_pickle_data(ConfigFilter.file_path, 'correct_uml_fid.pkl', correct_uml_fid)
    save_pickle_data(ConfigFilter.file_path, 'umls.pkl', umls_all)
    save_pickle_data(ConfigFilter.file_path, 'method2uml.pkl', method2uml_all)
