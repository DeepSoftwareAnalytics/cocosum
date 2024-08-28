import json
import time
from multiprocessing import cpu_count, Pool
from collections import Counter
import os
import sys
sys.path.append("../../")
from util.Config import Config as config
from util.DataUtil import read_pickle_data, split_and_filter_identifier, count_word, build_vocab_with_pad_unk, \
    build_vocab_with_pad_unk_sos_eos, extract_col, list_flatten, save_pickle_data, array_split, extract, \
    save_json_data, get_all_tokens


def build_vocab_info(code_word_frequency, summary_word_frequency, sbt_word_frequency):
    # build code vocab, include <NULL> and <UNK>. Start after <NULL>=0.
    code_w2i, code_i2w = build_vocab_with_pad_unk(code_word_frequency, 1, config.code_vocab_size - 2)
    # build summary vocab, include <s>, </s>, <NULL> and <UNK>. Start after </s>=2.
    summary_w2i, summary_i2w = build_vocab_with_pad_unk_sos_eos(summary_word_frequency, 3, config.summary_vocab_size - 4)
    # build code vocab, include <NULL> and <UNK>. Start after <NULL>=0.
    sbt_w2i, sbt_i2w = build_vocab_with_pad_unk(sbt_word_frequency, 1, config.sbt_vocab_size - 2)

    vocab = {'code_word_count': code_word_frequency,
             'summary_word_count': summary_word_frequency,
             'sbt_word_count': sbt_word_frequency,
             'summary_w2i': summary_w2i,
             'summary_i2w': summary_i2w,
             'sbt_i2w': sbt_i2w,
             'code_i2w': code_i2w,
             'code_w2i': code_w2i,
             'sbt_w2i': sbt_w2i}
    return vocab


def count_word_parallel(word_list):
    cores = cpu_count()
    pool = Pool(cores)
    word_split = array_split(word_list, cores)
    word_counts = pool.map(Counter, word_split)
    result = Counter()
    for word_count in word_counts:
        result += word_count
    pool.close()
    pool.join()
    return dict(result.most_common())  # return the dict sorted by frequency reversely.


def split_and_filter_identifier_parallel(seq):
    cores = cpu_count()
    pool = Pool(cores)
    seq_split = array_split(seq, cores)
    results = pool.map(split_and_filter_identifier, seq_split)
    pool.close()
    pool.join()
    return list_flatten(results)


class ConfigBuildVoc:
    root = "../../../Data/csn/"
    csn_data_path = os.path.join(root, "csn.pkl")
    # docstring_tokens_split_path = os.path.join(root, "docstring_tokens_split.pkl")
    # docstring_tokens_split_path = os.path.join(root, "summary/docstring_tokens_csn.pkl")
    docstring_tokens_split_path = os.path.join(root, "summary/docstring_tokens_csn.pkl")
    # code_tokens_javalang_split_path = os.path.join(root, "code_tokens_javalang_split.pkl")
    # code_tokens_csn_path = os.path.join(root, "code_tokens_csn.pkl")
    # code_tokens_javalang_split_path = os.path.join(root, "code_tokens_javalang_split_nofilter.pkl")
    # code_tokens_javalang = os.path.join(root, "code_tokens_javalang.pkl")
    # code_tokens_javalang_filter_punctuation = os.path.join(root, "code_tokens_javalang_filter_punctuation.pkl")
    # code_tokens_javalang_split_identifier = os.path.join(root, "code_tokens_javalang_split_identifier.pkl")
    # code_tokens_javalang_split_and_lower_identifier = os.path.join(root, "code_tokens_javalang_split_and_lower_identifier.pkl")
    code_tokens_javalang_replace_str_num= os.path.join(root, "code_tokens_javalang_replace_str_num.pkl")

    sbts_tokens_path = os.path.join(root, "sbts_tokens.pkl")


if __name__ == "__main__":

    # # calculate summary_word_count on trainset
    start = time.perf_counter()
    docstring_tokens = read_pickle_data(ConfigBuildVoc.docstring_tokens_split_path)
    # summary_word_list = split_and_filter_identifier_parallel(list_flatten(extract(docstring_tokens["train"])))
    summary_word_list = get_all_tokens(docstring_tokens["train"])
    print("split_and_filter_identifer_parallel - docstring_tokens: ", time.perf_counter() - start)

    start = time.perf_counter()
    summary_word_count = count_word_parallel(summary_word_list)
    print("count_word_parallel(summary_word_list): ", time.perf_counter() - start)

    save_pickle_data('../../../Data/csn/vocab_raw', 'summary_word_count.pkl', summary_word_count)
    print("summary_word_count length: ", len(summary_word_count))

    #calculate code_word_count on trainset
    start = time.perf_counter()
    # code_tokens_javalang = read_pickle_data(ConfigBuildVoc. code_tokens_javalang)
    # code_tokens_javalang = read_pickle_data(ConfigBuildVoc. code_tokens_javalang_filter_punctuation)
    # code_tokens_javalang = read_pickle_data(ConfigBuildVoc. code_tokens_javalang_split_identifier)
    # code_tokens_javalang = read_pickle_data(ConfigBuildVoc. code_tokens_javalang_split_and_lower_identifier)
    # code_tokens_javalang = read_pickle_data(ConfigBuildVoc. code_tokens_javalang_replace_str_num)
    code_tokens_javalang = read_pickle_data(ConfigBuildVoc. code_tokens_csn_path )
    # code_word_list = split_and_filter_identifier_parallel(list_flatten(extract(code_tokens_javalang["train"])))
    code_word_list = get_all_tokens(code_tokens_javalang["train"])
    print("split_and_filter_identifier_parallel - code_tokens: ", time.perf_counter() - start)

    start = time.perf_counter()
    code_word_count = count_word_parallel(code_word_list)
    print("count_word_parallel(code_word_list) time: ", time.perf_counter() - start)

    # save_pickle_data('../../../Data/csn/vocab_raw', 'filtered_code_word_count.pkl', code_word_count)
    save_pickle_data('../../../Data/csn/vocab_raw', 'code_word_count_csn.pkl', code_word_count)
    print("code_word_count length: ", len(code_word_count))

    # calculate sbts on trainset
    start = time.perf_counter()
    sbts_tokens = read_pickle_data(ConfigBuildVoc.sbts_tokens_path)
    sbt_word_list = get_all_tokens(sbts_tokens["train"])
    print("get sbt_tokens: ", time.perf_counter() - start)

    start = time.perf_counter()
    sbt_word_count = count_word_parallel(sbt_word_list)
    print("count_word_parallel(sbt_word_list) time: ", time.perf_counter() - start)

    save_pickle_data('../../../Data/csn/vocab_raw', 'filtered_sbt_word_count.pkl', sbt_word_count )
    print("sbt_word_count length: ", len(sbt_word_count))

    # build vocabulary from word counts
    summary_word_count = read_pickle_data(os.path.join('../../../Data/csn/vocab_raw', 'filtered_summary_word_count.pkl'))
    # code_word_count = read_pickle_data(os.path.join('../../../Data/csn/vocab_raw', 'filtered_code_word_count.pkl'))
    sbt_word_count = read_pickle_data(os.path.join('../../../Data/csn/vocab_raw', 'filtered_sbt_word_count.pkl'))
    start = time.perf_counter()
    # vocab_info = build_vocab_info(code_word_count, summary_word_count)
    vocab_info = build_vocab_info(code_word_count, summary_word_count, sbt_word_count)
    save_json_data('../../../Data/csn/vocab_raw', 'csn_trainingset_code_tokens_csn.json', vocab_info)
    print("build csn vocabulary: ", time.perf_counter() - start)
