# !/usr/bin/env python
# !-*-coding:utf-8 -*-
import time
import sys
import gzip
import os
import json
import pickle
sys.path.append("../../")
from util.Config import Config as config
from util.DataUtil import print_in_out_info, \
    build_vocab_with_pad_unk, build_vocab_with_pad_unk_sos_eos, transform_code, \
    read_data_and_count_word, transform_train_summary, transform_summary, save_pickle_data, print_dataset_info, \
    read_data_and_count_word_with_sbt, transform_sbt, label_encode_pkl_data


def read_train_data(data_dir):

    if config.using_sbt_flag:
        lines, code_word_count, summary_word_count, sbt_word_count = read_data_and_count_word_with_sbt(data_dir)
        sbt_w2i, sbt_i2w = build_vocab_with_pad_unk(sbt_word_count, 1, config.sbt_vocab_size - 2)
        sbt_unk_id = len(sbt_w2i) - 1
    else:
        lines, code_word_count, summary_word_count = read_data_and_count_word(data_dir)  # lines {}

    # build code vocab, include <NULL> and <UNK>. Start after <NULL>=0.
    code_w2i, code_i2w = build_vocab_with_pad_unk(code_word_count, 1, config.code_vocab_size - 2)
    code_unk_id = len(code_w2i) - 1

    # build summary vocab, include <s>, </s>, <NULL> and <UNK>. Start after </s>=2.
    summary_w2i, summary_i2w = build_vocab_with_pad_unk_sos_eos(summary_word_count, 3, config.summary_vocab_size - 4)
    summary_unk_id = len(summary_w2i) - 1
    summary_word_count[config.SOS_token], summary_word_count[config.EOS_token] = 0, 0

    # For summary, add SOS and EOS, and do padding to code_len
    method_code, method_summary, method_sbt= {}, {}, {}

    if config.is_pkl_data_flag:
        for m_id, line in lines.items():
            if config.using_sbt_flag:
                method_sbt[m_id] = transform_sbt(line, sbt_w2i, sbt_unk_id)
            method_code[m_id] = transform_code(line, code_w2i, code_unk_id)
            method_summary[m_id] = transform_summary(line, summary_w2i, summary_unk_id)

    else:
        for line in lines:
            if config.using_sbt_flag:
                # ToDo: add sbt
                pass
            method_code[len(method_code)] = transform_code(line, code_w2i, code_unk_id)
            summary_ids, summary_word_count = transform_train_summary(line, summary_w2i, summary_unk_id,
                                                                      summary_word_count)
            method_summary[len(method_summary)] = summary_ids

    if config.using_sbt_flag:
        return method_code, code_w2i, code_i2w, code_word_count, \
            method_summary, summary_w2i, summary_i2w, summary_word_count, \
            method_sbt, sbt_w2i, sbt_i2w, sbt_word_count
    else:
        return method_code, code_w2i, code_i2w, code_word_count, \
               method_summary, summary_w2i, summary_i2w, summary_word_count, \



# def read_test_data(file_path, code_w2i_dict, summary_w2i_dict):
def read_test_data(data_dir, code_w2i, summary_w2i, sbt_w2i={}):
    code_unk_id = code_w2i[config.UNK_token]
    summary_unk_id = summary_w2i[config.UNK_token]

    if config.is_pkl_data_flag:
        method_code, method_summary, method_sbt = \
            label_encode_pkl_data(data_dir, code_w2i, summary_w2i, code_unk_id, summary_unk_id, sbt_w2i)

    else:
        method_code, method_summary, method_sbt = \
            label_encode_pkl_data(data_dir, code_w2i, summary_w2i, code_unk_id, summary_unk_id, sbt_w2i)

    return method_code, method_summary, method_sbt


def process():
    train_code, code_w2i, code_i2w, train_code_word_count, \
     train_summary, summary_w2i, summary_i2w, train_summary_word_count, = read_train_data(config.train_data_path)

    val_code, val_summary, _ = read_test_data(config.val_data_path, code_w2i, summary_w2i)
    test_code, test_summary, _ = read_test_data(config.test_data_path, code_w2i, summary_w2i)

    data = {"ctrain": train_summary, "cval": val_summary, "ctest": test_summary,
            "dtrain": train_code, "dval": val_code, "dtest": test_code,
            "comstok": {"i2w": summary_i2w, "w2i": summary_w2i, "word_count": train_summary_word_count},
            "datstok": {"i2w": code_i2w, "w2i": code_w2i, "word_count": train_code_word_count},
            "config": {"datvocabsize": len(code_i2w), "comvocabsize": len(summary_i2w),
                       "datlen": config.code_len, "comlen": config.summary_len}}
    return data


def process_with_sbt():

    train_code, code_w2i, code_i2w, train_code_word_count, \
     train_summary, summary_w2i, summary_i2w, train_summary_word_count, \
     train_sbt, sbt_w2i, sbt_i2w, train_sbt_word_count = read_train_data(config.train_data_path)

    val_code, val_summary, val_sbt = read_test_data(config.val_data_path, code_w2i, summary_w2i, sbt_w2i)
    test_code, test_summary, test_sbt = read_test_data(config.test_data_path, code_w2i, summary_w2i, sbt_w2i)

    data = {"ctrain": train_summary, "cval": val_summary, "ctest": test_summary,
            "dtrain": train_code, "dval": val_code, "dtest": test_code,
            "strain": train_sbt, "sval": val_sbt, "stest": test_sbt,
            "comstok": {"i2w": summary_i2w, "w2i": summary_w2i, "word_count": train_summary_word_count},
            "datstok": {"i2w": code_i2w, "w2i": code_w2i, "word_count": train_code_word_count},
            "smlstok": {"i2w": sbt_i2w, "w2i": sbt_w2i, "word_count": train_sbt_word_count},
            "config": {"datvocabsize": len(code_i2w), "comvocabsize": len(summary_i2w), "smlvocabsize": len(sbt_i2w),
                       "datlen": config.code_len, "comlen": config.summary_len, "smllen": config.sbt_len}}
    return data


if __name__ == "__main__":
    print_in_out_info()

    start = time.perf_counter()
    if config.using_sbt_flag:
        dataset = process_with_sbt()
        save_pickle_data(config.data_dir, 'dataset_with_sbt.pkl', dataset)
    else:
        dataset = process()
        save_pickle_data(config.data_dir, 'dataset_no_sbt.pkl', dataset)
    end = time.perf_counter()
    print("time: ", end - start)

    print_dataset_info(dataset)
