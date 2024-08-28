# -*- coding: utf-8 -*-
# @Time      : 2020-02-21 16:33
# @Author    : Eason Hu
# @Site      :
# @File      : get_sbt_csn.py

import time
import os
import pickle
import traceback
from util.DataUtil import make_directory, sbt_parser
import json
from multiprocessing import cpu_count, Pool
import numpy as np
from util.DataUtil import read_data_append_id
from functools import partial

def parallelize(data, func):
    sbts = {}
    cores = cpu_count()
    pool = Pool(cores)
    data_split = np.array_split(data, cores)
    method_data = pool.map(func, data_split)
    for sbt in method_data:
        sbts.update(sbt)
    pool.close()
    pool.join()
    return sbts

def process_lines(lines):
    sbts = {}
    for method in lines:
        try:
            java_file_path = write_source_code_to_java_file(method['id'], method['code'])
            sbts[method['id']] = sbt_parser(java_file_path)
        except:
            traceback.print_exc(file=open("error.txt", 'a'))
            print("error_id:", method['id'])
            continue
    return sbts

# Write to java file
def write_source_code_to_java_file(method_id, method):
    java_path = os.path.join(os.getcwd(), r"java_file", str(method_id)+".java")
    if os.path.isfile(java_path):
        return java_path
    with open(java_path, "w") as f:
        f.write(method)
    return java_path


# Get the sbt of the method's source code
def obtain_sbt(data_dir, sbt_file_path):
    data = json.load(open(data_dir, 'r'))
    lines = []
    for fid in data:
        lines.append({'id':fid, 'code':data[fid]})
    # lines = read_data_append_id(os.path.join(data_dir))
    print('start process sbt tree')
    start_time = time.perf_counter()
    sbt_data = parallelize(lines, process_lines)

    print("time cost of obtaining sbt :", time.perf_counter() - start_time)
    with open(os.path.join(sbt_file_path, 'sbts_{}.pkl'), 'wb') as f:
        pickle.dump(sbt_data, f)
    print("sbts count", len(sbt_data))


if __name__ == '__main__':
    methods_path = '/datadrive/yuxuan_data/funcom/funcom_processed/functions.json' # Azure52
    # methods_path =  'F:/msra/dataset/csn/java/final/jsonl/'  # window
        # make_directory(os.path.join(os.getcwd(), r"java_file"))  # write source code to java in this path
    output_path = '../../data/sbt_leclair_by_file'
    make_directory(os.path.join(output_path, 'java_file'))  # write source code to java in this path
    obtain_sbt(methods_path, output_path)
