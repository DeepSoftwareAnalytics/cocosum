# !/usr/bin/env python
# !-*-coding:utf-8 -*-
import os
import sys
import time
from multiprocessing import Pool

import natsort

sys.path.append("../../")
from util.DataUtil import make_directory, print_time
from datapreprocess.get_uml import obtain_uml


class Config(object):

    uml_graph_cmd_opt = r" -inferrel -inferreltype navassoc -collpackages java.util.* -inferdep -inferdepinpackage  " \
                        r"-hide java.*  -all -private  -output "
    uml_graph_cmd = r"java -jar  /usr/local/lib/UmlGraph.jar  -sourcepath  "

    # Azure52
    # methods_dir = "/datadrive/enshi/split_data/train/"
    # repositories_dir = r"/datadrive/Data/csn/github_repositories/train/"
    # sha2repository_name = os.path.join(repositories_dir, "name_list.pkl")
    #
    # save_dot_dir = r"/datadrive/enshi/dot/20200310/train/"
    # save_uml_dir = r"/datadrive/enshi/uml/mini_data/train/"
    #
    # save_error_dir = r"/datadrive/enshi/error/20200308/train/"

    # Azure 13
    methods_dir = "/mnt/enshi/split_data/train/"
    repositories_dir = r"/mnt/Data/csn/github_repositories/train/"
    sha2repository_name = os.path.join(repositories_dir, "name_list.pkl")

    save_dot_dir = r"/mnt/enshi/dot/20200308/train/"
    save_uml_dir = r"/mnt/enshi/uml/mini_data/train/"

    save_error_dir = r"/mnt/enshi/error/train/"

    # all_file_number = 152

    # # Azure52
    # start_number = 0
    # processing_number = 20

    # Azure157
    # start_number = 120
    # processing_number = 12

    # Azure157
    start_number = 40
    processing_number = 12

    # # Azure13
    # start_number = 60
    # processing_number = 20


if __name__ == '__main__':

    start_time = time.perf_counter()

    make_directory(Config.save_dot_dir)
    make_directory(Config.save_uml_dir)
    make_directory(Config.save_error_dir)

    data_directory = Config.methods_dir
    new_data_name = os.listdir(data_directory)
    new_data_name = natsort.natsorted(new_data_name)
    # print( new_data_name )

    old_data_path = [os.path.join(data_directory, file_name) for file_name in new_data_name]
    print("files count", len(old_data_path))
    processing_number = Config.processing_number
    # Multiprocessing  https://docs.python.org/2/library/multiprocessing.html
    # start_file_number = 0
    end_file_number = Config.start_number + 2 * Config.processing_number + 1
    for start_file_number in range(Config.start_number, end_file_number, Config.processing_number):
        p = Pool(processing_number)
        for idx in range(start_file_number, start_file_number + processing_number):
            # if idx < len(old_data_path):
            if idx < 60:
                p.apply_async(obtain_uml, args=(old_data_path[idx], Config.sha2repository_name, Config.repositories_dir,
                                                Config.uml_graph_cmd, Config.uml_graph_cmd_opt, Config.save_dot_dir,
                                                new_data_name[idx], Config.save_uml_dir, Config.save_error_dir, idx))
        p.close()  # Indicate that no more data will be put on this queue by the current process.
        p.join()   # Block until all items in the queue have been gotten and processed.

    # for test
    # idx = 0
    # obtain_uml(old_data_path[idx], Config.sha2repository_name, Config.repositories_dir,
    #            Config.uml_graph_cmd, Config.uml_graph_cmd_opt, Config.save_dot_dir,
    #            new_data_name[idx], Config.save_uml_dir, Config.save_error_dir, idx)
    # # Print time cost
    print_time(time.perf_counter() - start_time)
