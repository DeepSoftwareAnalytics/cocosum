# !/usr/bin/env python
# !-*-coding:utf-8 -*-

import os
import pickle
import sys
import time

sys.path.append("../../")
from util.DataUtil import print_time, save_pickle_data


def merge_data(source, destination):
    methods = {}
    umls = {}
    method2uml_index = {}
    sbts = {}

    for r, d, f in os.walk(source):
        for file_name in f:
            if file_name.endswith("methods.pkl"):
                methods.update(pickle.load(open(os.path.join(r, file_name), "rb")))
            if file_name.endswith("umls.pkl"):
                umls.update(pickle.load(open(os.path.join(r, file_name), "rb")))
            if file_name.endswith("method2uml_index.pkl"):
                method2uml_index.update(pickle.load(open(os.path.join(r, file_name), "rb")))
            if file_name.endswith("sbts.pkl"):
                sbts.update(pickle.load(open(os.path.join(r, file_name), "rb")))

    print("length of methods ", len(methods))
    print("length of umls ", len(umls))
    print("length of method2uml_index ", len(method2uml_index))
    # print("length of sbts ", len(sbts))

    save_pickle_data(destination, 'methods.pkl', methods)
    save_pickle_data(destination, 'umls.pkl', umls)
    save_pickle_data(destination, 'method2uml_index.pkl', method2uml_index)
    # write_to_pickle(destination, sbts, file_name='sbts.pkl', )


class ConfigMerge:
    # test on window
    # load_path = r"D:\MSRA\yanlcodesum"
    # save_path = r"D:\MSRA\yanlcodesum\all"
    load_path = r"/mnt/enshi/uml/mini_data/"
    save_path = r"/mnt/enshi/uml/all_data_uml/"


if __name__ == '__main__':

    # parser = argparse.ArgumentParser()
    # parser.add_argument("-p", dest='data_dir', choices=['train', 'valid', 'test'], type=str, default='valid')
    # args = parser.parse_args()
    # dataset_dir = args.data_dir
    dataset_dirs = ['train', 'valid', 'test']
    for dataset_dir in dataset_dirs:
        print(dataset_dir)
        start_time = time.perf_counter()
        source_path = os.path.join(ConfigMerge.load_path, dataset_dir)
        destination_path = os.path.join(ConfigMerge.save_path, dataset_dir)

        merge_data(source_path, destination_path)
        print_time(time.perf_counter() - start_time)

