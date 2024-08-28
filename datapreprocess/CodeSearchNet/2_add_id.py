import gzip
import json
import os
import sys
import natsort
import time
sys.path.append("../../")
from util.DataUtil import save_pickle_data, time_format
from util.Config import Config as cf
from util.LoggerUtil import set_logger, debug_logger


def add_id(data_dir, mid):
    data = {}
    files_name = []
    for r, d, f in os.walk(data_dir):
        for file_name in f:
            if file_name.endswith(".jsonl.gz"):
                files_name.append(os.path.join(r, file_name))
    files_name = natsort.natsorted(files_name)
    for f in files_name:
        debug_logger(f)
        with gzip.open(f, 'r') as file:
            for line in file:
                line = json.loads(line)
                data[mid] = line
                mid += 1
    return mid, data


def csn_add_fid():
    start_id = 0
    data_with_fid = {}
    dataset_partitions = ['train', 'valid', 'test']
    for partitions in dataset_partitions:
        debug_logger(partitions)
        data_dir = os.path.join(Config.csn_ori_java_path, partitions)
        start_id, data = add_id(data_dir, start_id)
        data_with_fid[partitions] = data
    return data_with_fid


class Config:
    dataset_path = '../../data/csn/'  # TODO. move 157 data to teamdrive.
    csn_ori_java_path = os.path.join(dataset_path, "java/final/jsonl/")  # TODO. move to teamdrive.


if __name__ == '__main__':
    # TODO: skip this step if csn.pkl already exits.
    set_logger(cf.DEBUG)
    start = time.perf_counter()
    data = csn_add_fid()
    save_pickle_data(Config.dataset_path, 'csn.pkl', data)
    debug_logger(" time cost :" + time_format(time.perf_counter() - start))
