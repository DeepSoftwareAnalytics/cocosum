# !/usr/bin/env python
# !-*-coding:utf-8 -*-
import json
import os
import time
import traceback
import urllib.request
import zipfile
import gzip
import sys
sys.path.append("../../")
from util.DataUtil import make_directory, time_format, save_pickle_data
from util.LoggerUtil import set_logger, debug_logger
from multiprocessing import cpu_count, Pool
from util.Config import Config as cf


def extract_url_and_name(file):
    """
     Extract url and name of github repository according to the file
     :return
     urls: dict like {sha:url...}
     names: dict like {sha:git_repo_name,...}
     """
    urls, names = {}, {}
    debug_logger("processing %s " % file)
    with gzip.open(file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        method = json.loads(line)
        repo, sha = method["repo"], method["sha"]
        if sha in names.keys():
            continue
        url = 'https://github.com/' + repo + '/' + 'archive/' + sha + '.zip'
        name = repo.split("/")[-1]
        urls[sha],  names[sha] = url, name
    return urls, names


def download_repository(item):
    ((sha, url), (_, name), save_dir) = item
    """
    given the url download the repository and save in " save_dir + names[sha]"
    :param urls: dict like {sha:url...}
    :param names: dict like {sha:git_repo_name,...}
    :param save_dir:
    """
    if os.path.exists(os.path.join(save_dir, name + "-" + sha)):
        pass
    else:
        # noinspection PyBroadException
        try:
            save_path = os.path.join(save_dir, name + '.zip')
            urllib.request.urlretrieve(url, save_path)
        except Exception:
            traceback.print_exc(file=open("download_repository.txt", 'a'))


def unzip(path):
    """
    Unzip the file in the path
    """
    os.chdir(path)
    for file_name in os.listdir(path):
        if os.path.splitext(file_name)[-1] == '.zip':
            # noinspection PyBroadException
            try:
                file_zip = zipfile.ZipFile(file_name, 'r')
                for file in file_zip.namelist():
                    file_zip.extract(file, path)
                file_zip.close()
                os.remove(file_name)
            except Exception:
                traceback.print_exc(file=open("unzip_error.txt", 'a'))
                continue


def extract_url_and_name_all(path):
    """
     Extract url and name of github repository
     :return
     urls: dict like {sha:url...}
     names: dict like {sha:git_repo_name,...}
     """
    sha2url = {}  # {sha:url,...}
    sha2repo_name = {}  # {sha:repository_name,...}
    for r, d, f in os.walk(path):
        for file_name in f:
            if file_name.endswith(".jsonl.gz"):
                sha2url_part, sha2repo_name_part = extract_url_and_name(os.path.join(r, file_name))
                sha2url.update(sha2url_part)
                sha2repo_name.update(sha2repo_name_part)

    return sha2url, sha2repo_name


def get_git_repository_parallel(load_data_dir, save_repo_dir):
    """
    download and save git repos in save_repo_dir according to all json files in load_data_dir.
    In the same time, generate sha2url.pkl sha2repoName.pkl.
    """
    sha2url, sha2repo_name = extract_url_and_name_all(load_data_dir)
    cores = cpu_count()
    pool = Pool(cores)
    x = zip(sha2url.items(), sha2repo_name.items(), [save_repo_dir] * len(sha2url))
    pool.map(download_repository, x)
    debug_logger("len(sha2repo_name) %d  " % len(sha2repo_name))
    save_pickle_data(save_repo_dir, "sha2url.pkl", sha2url)
    save_pickle_data(save_repo_dir, "sha2repoName.pkl", sha2repo_name)
    unzip(save_repo_dir)


class Config:

    root_path = os.path.dirname(os.path.realpath(".."))
    csn_data_dir = os.path.join(root_path, "data/csn/java/final/jsonl")
    repository_dir = os.path.join(root_path, "data/csn/github_repositories")


if __name__ == '__main__':
    # TODO: skip this step if all repo already exits.
    set_logger(cf.DEBUG)
    parts = ['train', 'valid', 'test']
    start = time.perf_counter()
    for part in parts:
        original_csn_data_path = os.path.join(Config.csn_data_dir, part)
        git_repository_path = os.path.join(Config.repository_dir, part)

        make_directory(git_repository_path)
        get_git_repository_parallel(original_csn_data_path, git_repository_path)

        debug_logger(part + " time cost :" + time_format(time.perf_counter() - start))
        start = time.perf_counter()
