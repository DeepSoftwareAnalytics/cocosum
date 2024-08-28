import sys

sys.path.append("../../")
import os
import gzip
import json
from util.Config import Config as config
from util.DataUtil import print_in_out_info, \
    build_vocab_with_pad_unk, build_vocab_with_pad_unk_sos_eos, transform_code, \
    read_data_and_count_word, transform_train_summary, transform_summary, save_pickle_data, print_dataset_info
import time


"""
Convert original CodeSearchNet data (without SBT) to the format we use 
"""


def read_train_data(data_dir):
    lines, code_word_count, summary_word_count = read_data_and_count_word(data_dir)
    # build code vocab, include <NULL> and <UNK>. Start after <NULL>=0.
    code_w2i, code_i2w = build_vocab_with_pad_unk(code_word_count, 1, config.code_vocab_size - 2)
    code_unk_id = len(code_w2i) - 1
    # build summary vocab, include <s>, </s>, <NULL> and <UNK>. Start after </s>=2.
    summary_w2i, summary_i2w = build_vocab_with_pad_unk_sos_eos(summary_word_count, 3, config.summary_vocab_size - 4)
    summary_unk_id = len(summary_w2i) - 1

    summary_word_count[config.SOS_token] = 0
    summary_word_count[config.EOS_token] = 0

    # map each token to an int id
    # For summary, add SOS and EOS, and do padding to token_len
    method_code = {}
    method_summary = {}

    for line in lines:
        code_ids = transform_code(line, code_w2i, code_unk_id)
        method_code[len(method_code)] = code_ids
        summary_ids, summary_word_count = transform_train_summary(line, summary_w2i, summary_unk_id, summary_word_count)
        method_summary[len(method_summary)] = summary_ids

    return method_code, code_w2i, code_i2w, code_word_count, method_summary, summary_w2i, summary_i2w, summary_word_count


def read_test_data(data_dir, code_w2i, summary_w2i):
    method_code = {}
    method_summary = {}

    code_unk_id = code_w2i[config.UNK_token]
    summary_unk_id = summary_w2i[config.UNK_token]

    for r, d, f in os.walk(data_dir):
        for file_name in f:
            if file_name.endswith(".jsonl.gz"):
                with gzip.open(os.path.join(r, file_name), 'r') as f:
                    print(os.path.join(r, file_name))
                    for line in f:
                        line = json.loads(line)
                        # If the token is not in the token map built on train set, we will use UNK token to indicate it
                        code_ids = transform_code(line, code_w2i, code_unk_id)
                        method_code[len(method_code)] = code_ids
                        summary_ids = transform_summary(line, summary_w2i, summary_unk_id)
                        method_summary[len(method_summary)] = summary_ids
                        # # TODO: For test: check the <UNK> word
                        # for token in split_identifier(line["docstring_tokens"])[:config.summary_len - 1]:
                        #     if token in summary_w2i:
                        #         summary_ids.append(summary_w2i[token])
                        #     else:
                        #         print("UNK token: ", token)
                        #         exit(-1)
    return method_code, method_summary


def process():
    dtrain, code_w2i, code_i2w, train_code_word_count, ctrain, summary_w2i, summary_i2w, train_summary_word_count = read_train_data(config.train_data_path)
    val_method_code, val_method_summary = read_test_data(config.val_data_path, code_w2i, summary_w2i)
    test_method_code, test_method_summary = read_test_data(config.test_data_path, code_w2i, summary_w2i)

    dataset = {"ctrain": ctrain, "cval": val_method_summary, "ctest": test_method_summary,
               "dtrain": dtrain, "dval": val_method_code, "dtest": test_method_code,
               "comstok": {"i2w": summary_i2w, "w2i": summary_w2i, "word_count": train_summary_word_count},
               "datstok": {"i2w": code_i2w, "w2i": code_w2i, "word_count": train_code_word_count},
               "config": {"datvocabsize": len(code_i2w), "comvocabsize": len(summary_i2w),
                          "datlen": config.code_len, "comlen": config.summary_len}}
    return dataset


if __name__ == "__main__":
    print_in_out_info()
    start = time.perf_counter()
    dataset = process()
    save_pickle_data(config.data_dir, 'standard_data.pkl', dataset)
    end = time.perf_counter()
    print("time: ", end - start)
    print_dataset_info(dataset)