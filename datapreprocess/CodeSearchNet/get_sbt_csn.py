# -*- coding: utf-8 -*-
# @Time      : 2020-02-21 16:33
# @Author    : Eason Hu
# @Site      :
# @File      : get_sbt_csn.py

import time
import os
import pickle
import traceback
from multiprocessing import cpu_count, Pool
import sys

sys.path.append("../../")
from util.DataUtil import make_directory, sbt_parser, write_source_code_to_java_file, array_split


# Get the sbt of the method's source code
def obtain_sbt(method_file_path, sbt_file_path):
    sbts = {}  # dict {method_id1 : sbt_1 , method_id2 : sbt_2,...}

    error_count = 0
    print('start process sbt tree')
    start_time = time.perf_counter()

    methods = pickle.load(open(method_file_path, "rb"))
    for idx, method in methods.items():
        print(idx)
        try:
            java_file_path = write_source_code_to_java_file("./", idx, method["code"])
            sbts[idx] = sbt_parser(java_file_path)
        except:
            traceback.print_exc(file=open("error.txt", 'a'))
            error_count += 1
            continue

    print("time cost of obtaining sbt :", time.perf_counter() - start_time)
    with open(os.path.join(sbt_file_path, 'sbts.pkl'), 'wb') as f:
        pickle.dump(sbts, f)
    print("error_count count", error_count)
    print("sbts count", len(sbts))


def write_code_to_java_file(method):
    """
    write code to a java file
    :param method: (fid, code_string)
    :return:
    """
    make_directory(ConfigSbt.java_files)
    write_source_code_to_java_file(ConfigSbt.java_files, method[0], method[1])


def write_source_code_to_java_file_list(methods):
    """
    write source code to java files.
    :param methods: a list of paris [(fid, code_string)]
    :return:
    """
    for method in methods:
        write_source_code_to_java_file(ConfigSbt.java_files, method[0], method[1])


def get_sbt_str(method_pair):
    """
    Given (fid, code_str), return sbt_str
    """
    return sbt_parser(os.path.join(ConfigSbt.java_files, str(method_pair[0]) + ".java"))


def sbt_parser_lists(methods):
    sbts = {}
    for idx, method in methods:
        try:
            sbts[idx] = sbt_parser(os.path.join(ConfigSbt.java_files, str(idx) + ".java"))
        except:
            traceback.print_exc(file=open("error.txt", 'a'))
            continue
    return sbts


def gen_sbt_parallel(data):
    methods_src = {}
    for idx, item in data.items():
        methods_src[idx] = item["code"]

    cores = cpu_count()
    data_split = array_split(list(methods_src.items()), cores)
    # write source code to .java file
    make_directory(ConfigSbt.java_files)
    pool = Pool(cores)
    pool.map(write_source_code_to_java_file_list, data_split)
    pool.close()
    pool.join()

    # parser .java file to obtain sbt
    pool = Pool(cores)
    results = pool.map(sbt_parser_lists, data_split)
    pool.close()
    pool.join()

    sbts = {}
    for sbt in results:
        sbts.update(sbt)
    return sbts


class ConfigSbt:
    java_files = "../../../Data/java_files"
    # java_files = "/mnt/xiaodi/cocogum_refactor/csn_mini_data/java_files"
    # java_files = r"D:\MSRA\yanlcodesum\Data\java_files"  # window


if __name__ == '__main__':
    # methods_path = '/mnt/enshi/CodeSearchNet/uml/1k' # Azure52
    methods_path = r'D:\MSRA\yanlcodesum\1k\valid\methods.pkl'  # window
    make_directory(ConfigSbt.java_files)  # write source code to java in this path
    # obtain_sbt(os.path.join(methods_path, "method_class_in_package.pkl"), methods_path)
    train_data = pickle.load(open(methods_path, "rb"))
    sbts = gen_sbt_parallel(train_data)
