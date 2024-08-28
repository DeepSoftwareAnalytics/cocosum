# !/usr/bin/env python
# !-*-coding:utf-8 -*-

import sys
import natsort
import os
import subprocess
import time
import traceback
from multiprocessing import cpu_count, Pool
sys.path.append("../../")
from util.Config import Config as cf
from util.LoggerUtil import set_logger, debug_logger
from util.DataUtil import make_directory, time_format, read_pickle_data, save_pickle_data


def get_package_name(path_of_java_file):
    package = ""
    with open(path_of_java_file) as file:
        lines = file.readlines()
    for line in lines:
        if "package" in line[:7]:
            package = line.replace("package", "")
            package = package.replace(";\n", "").replace(";", "").replace(" ", "")
            break
    return package


def get_package_absolute_path(java_fun_path, package_name):
    package_relative_path = package_name.replace('.', "/")
    pos = java_fun_path.find(package_relative_path)
    package_path = java_fun_path[:pos]
    return package_path


def get_java_path(repository_name, method):
    file_path = os.path.normpath(repository_name + "-" + method["sha"] + "/" + method["path"])
    java_file_path = os.path.join(Config.repositories_dir, file_path)
    return java_file_path


def parser_package_using_umlgraph(repository_name, package_name, method, fid):
    package_path = get_package_absolute_path(get_java_path(repository_name, method), package_name)
    output_file_path = os.path.join(Config.save_dot_dir, str(fid) + ".dot")
    output_name = Config.uml_graph_cmd_opt + output_file_path
    subprocess.getstatusoutput(Config.uml_graph_cmd + package_path + "  " + package_name + output_name)


def gen_uml_dot_file(path):
    start = time.perf_counter()
    debug_logger(path.split("/")[-1])
    m2uid, prev_method_path, prev_repo_sha, prev_package_name, prev_fid = {}, "", "", "", 0

    sha2repository = read_pickle_data(Config.sha2repoName_path)
    small_dataset = read_pickle_data(path)
    error_file = Config.save_error_dir + path.split("/")[-1].split(".")[0] + "_error.txt"

    for fid, method in small_dataset.items():
        if method["path"] == prev_method_path:
            m2uid[fid] = prev_fid
            continue
        try:
            repository_name = sha2repository[method["sha"]]
            package_name = get_package_name(get_java_path(repository_name, method))
            if prev_repo_sha == method["sha"] and prev_package_name == package_name:
                m2uid[fid] = prev_fid
                continue
            parser_package_using_umlgraph(repository_name, package_name, method, fid)
            prev_repo_sha, prev_package_name, prev_method_path = method["sha"], package_name, method["path"]
            m2uid[fid] = fid
            prev_fid = fid
        except:
            traceback.print_exc(file=open(error_file, 'a'))

        # if fid > 10:
        #     break
    debug_logger(path.split("/")[-1] + "  time cost : %s " % time_format(time.perf_counter() - start))
    save_pickle_data(Config.save_m2uid_dir, path.split("/")[-1].split(".")[0] + '_m2uid.pkl', m2uid)


class Config(object):
    part = "train"
    uml_graph_cmd_opt = " -inferrel -inferreltype navassoc -collpackages java.util.* -inferdep -inferdepinpackage  " \
                        "-hide java.*  -all -private  -output "
    uml_graph_cmd = "java -jar  UmlGraph.jar  -sourcepath  "
    splitted_data_dir = "../../data/csn/split_data/"
    repositories_dir = "../../data/csn/github_repositories/"
    sha2repoName_path = os.path.join(repositories_dir, "sha2repoName.pkl")
    save_dot_dir = "../../data/csn/uml/dot/"
    save_uml_dir = "../../data/csn/uml/"
    save_error_dir = "../../data/csn/uml/error/"
    save_m2uid_dir = "../../data/csn/uml/m2uid/"


def make_dir():
    make_directory(Config.save_dot_dir)
    make_directory(Config.save_uml_dir)
    make_directory(Config.save_error_dir)
    make_directory(Config.save_m2uid_dir)


def set_config():
    Config.splitted_data_dir = os.path.join("../../data/csn/split_data/", Config.part)
    Config.repositories_dir = os.path.join("../../data/csn/github_repositories/", Config.part)
    Config.sha2repoName_path = os.path.join(Config.repositories_dir, "sha2repoName.pkl")
    Config.save_dot_dir = os.path.join("../../data/csn/uml/dot/", Config.part)
    Config.save_uml_dir = os.path.join("../../data/csn/uml/", Config.part)
    Config.save_error_dir = os.path.join("../../data/csn/uml/error/", Config.part)
    Config.save_m2uid_dir = os.path.join("../../data/csn/uml/m2uid/", Config.part)


if __name__ == '__main__':
    set_logger(cf.DEBUG)
    start_time = time.perf_counter()
    parts = ['train', 'valid', 'test']
    cores = cpu_count()

    for part in parts:
        Config.part = part
        set_config()
        make_dir()
        small_data_name = natsort.natsorted(os.listdir(Config.splitted_data_dir))
        all_data_path = [os.path.join(Config.splitted_data_dir, file_name) for file_name in small_data_name]
        debug_logger("files count %d" % len(all_data_path))
        pool = Pool(cores)
        pool.map(gen_uml_dot_file, all_data_path)
        pool.close()
        pool.join()
        # gen_uml_dot_file(all_data_path[0])
        # break
    debug_logger("Time cost %s" % time_format(time.perf_counter() - start_time))
