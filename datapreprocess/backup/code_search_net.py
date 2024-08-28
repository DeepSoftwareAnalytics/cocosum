# !/usr/bin/env python
# !-*-coding:utf-8 -*-
import sys

sys.path.append("..")
import os
import json
from util.Config import Config as config
from util.DataUtil import padding
import pickle
import time
from spiral import ronin
import string

"""
reset origin version about this file
"""
# Identifier splitting (also known as identifier name tokenization) is the task of breaking apart program identifier
# strings such as getInt or readUTF8stream into component tokens: [get, int] and [read, utf8, stream].
def split_identifier(sequence):
    toks = []
    for s in sequence:
        if s in string.punctuation:
            toks.append(s)
        else:
            toks.extend([tok.lower() for tok in ronin.split(s)])
    return toks


def count_word(token_word_count, tokens):
    for token in tokens:
        token_word_count[token] = token_word_count.get(token, 0) + 1
    return token_word_count


# if vocab_size == -1, all tokens will be used
def build_vocab(word_to_ix, value_to_ix, ix_to_value, max_vocab_size=-1):
    L = sorted(word_to_ix.items(), key=lambda item: item[1], reverse=True)

    if max_vocab_size != -1:
        print("vocab_size ", max_vocab_size)
        size = max_vocab_size
    else:
        print("use all tokens")
        size = len(L)

    for i in range(size):
        value_to_ix[L[i][0]] = len(value_to_ix)
        ix_to_value[value_to_ix[L[i][0]]] = L[i][0]
    return value_to_ix, ix_to_value


def read_train_data(file_path):
    # initial vocabulary, index2word and word count
    tok_map_dict = {"<Padding>": config.PAD_token_id, "<SOS>": config.SOS_token_id, "<EOS>": config.EOS_token_id,
                    "<UNK>": config.UNK_token_id}
    tok_i2w = {config.PAD_token_id: "<Padding>", config.SOS_token_id: "<SOS>", config.EOS_token_id: "<EOS>",
               config.UNK_token_id: "<UNK>"}
    tok_word_count = {}

    sum_map_dict = {"<Padding>": config.PAD_token_id, "<SOS>": config.SOS_token_id, "<EOS>": config.EOS_token_id,
                    "<UNK>": config.UNK_token_id}
    sum_i2w = {config.PAD_token_id: "<Padding>", config.SOS_token_id: "<SOS>", config.EOS_token_id: "<EOS>",
               config.UNK_token_id: "<UNK>"}
    sum_word_count = {}

    sbt_map_dict = {"<Padding>": config.PAD_token_id, "<SOS>": config.SOS_token_id, "<EOS>": config.EOS_token_id,
                    "<UNK>": config.UNK_token_id}
    sbt_i2w = {config.PAD_token_id: "<Padding>", config.SOS_token_id: "<SOS>", config.EOS_token_id: "<EOS>",
               config.UNK_token_id: "<UNK>"}
    sbt_word_count = {}

    lines = []

    # count word frequency
    for r, d, f in os.walk(file_path):
        for file_name in f:
            if file_name.endswith(".jsonl"):
                with open(os.path.join(r, file_name), 'r') as f:
                    sample_files = f.readlines()
                    print(os.path.join(r, file_name))
                    sample_files = json.loads(sample_files[0])
                    for line in sample_files:
                        tok_word_count = count_word(tok_word_count,
                                                    split_identifier(line["code_tokens"])[:config.code_len])  # query
                        sum_word_count = count_word(sum_word_count,
                                                    split_identifier(line["docstring_tokens"])[
                                                    :config.summary_len])  # answer
                        sbt_word_count = count_word(sbt_word_count, line["SBT"][:config.sbt_len])  # answer
                        lines.append(line)

    # build vocabulary
    tok_map_dict, tok_i2w = build_vocab(tok_word_count, tok_map_dict, tok_i2w, config.code_vocab_size)
    sum_map_dict, sum_i2w = build_vocab(sum_word_count, sum_map_dict, sum_i2w, config.summary_vocab_size)
    sbt_map_dict, sbt_i2w = build_vocab(sbt_word_count, sbt_map_dict, sbt_i2w, config.sbt_vocab_max_size)

    # map each token to an int id,   add EOS_Token  , Padding to token_len
    summaries = []
    methods_token = []
    methods_sbt = []

    for line in lines:
        # ls = [token for token in line["code_tokens"][:cf.token_len - 1]]
        mapped_method_token_ids = [tok_map_dict.get(token.lower(), config.UNK_token_id) for token in
                                   split_identifier(line["code_tokens"])[:config.code_len - 1]]
        mapped_method_token_ids.append(config.EOS_token_id)
        mapped_method_token_ids = padding(mapped_method_token_ids, config.code_len, config.PAD_token_id)
        methods_token.append(mapped_method_token_ids)

        mapped_sum_ids = [sum_map_dict.get(token.lower(), config.UNK_token_id) for token in
                          split_identifier(line["docstring_tokens"])[:config.summary_len - 1]]
        mapped_sum_ids.append(config.EOS_token_id)
        mapped_sum_ids = padding(mapped_sum_ids, config.summary_len, config.PAD_token_id)
        summaries.append(mapped_sum_ids)

        mapped_method_sbt_ids = [sbt_map_dict.get(token.lower(), config.UNK_token_id) for token in
                                 line["SBT"][:config.sbt_len - 1]]
        mapped_method_sbt_ids.append(config.EOS_token_id)
        mapped_method_sbt_ids = padding(mapped_method_sbt_ids, config.sbt_len, config.PAD_token_id)
        methods_sbt.append(mapped_method_sbt_ids)

    # calculate the size of vocabulary
    tok_vocab_size = len(tok_map_dict)
    sum_vocab_size = len(sum_map_dict)
    sbt_vocab_size = len(sbt_map_dict)
    print("Read train data: method-summary pair %d,  vocab_size %d ,sbt_vocab_size %d ,summary_vocab_size %d " % (
        len(summaries), tok_vocab_size, sbt_vocab_size, sum_vocab_size))

    return summaries, methods_token, methods_sbt, tok_map_dict, sum_map_dict, sbt_map_dict, tok_word_count, sum_word_count, sbt_word_count, tok_i2w, sum_i2w, sbt_i2w


def read_test_data(file_path, tok_map_dict, sum_map_dict, sbt_map_dict):
    # map each token to an int id,   add EOS_Token  , Padding to token_len
    summaries = []
    methods_token = []
    methods_sbt = []

    # count word frequency
    for r, d, f in os.walk(file_path):
        for file_name in f:
            if file_name.endswith(".jsonl"):
                with open(os.path.join(r, file_name), 'r') as f:
                    sample_files = f.readlines()
                    print(os.path.join(r, file_name))
                    sample_files = json.loads(sample_files[0])
                    for line in sample_files:
                        # If the token is not in the token map built on train set, we will use UNK token to indicate it
                        mapped_method_token_ids = [tok_map_dict.get(token.lower(), config.UNK_token_id) for token in
                                                   split_identifier(line["code_tokens"])[:config.code_len - 1]]
                        mapped_method_token_ids.append(config.EOS_token_id)
                        mapped_method_token_ids = padding(mapped_method_token_ids, config.code_len, config.PAD_token_id)
                        methods_token.append(mapped_method_token_ids)

                        mapped_sum_ids = [sum_map_dict.get(token.lower(), config.UNK_token_id) for token in
                                          split_identifier(line["docstring_tokens"])[:config.summary_len - 1]]
                        mapped_sum_ids.append(config.EOS_token_id)
                        mapped_sum_ids = padding(mapped_sum_ids, config.summary_len, config.PAD_token_id)
                        summaries.append(mapped_sum_ids)

                        mapped_method_sbt_ids = [sbt_map_dict.get(token.lower(), config.UNK_token_id) for token in
                                                 line["SBT"][:config.sbt_len - 1]]
                        mapped_method_sbt_ids.append(config.EOS_token_id)
                        mapped_method_sbt_ids = padding(mapped_method_sbt_ids, config.sbt_len, config.PAD_token_id)
                        methods_sbt.append(mapped_method_sbt_ids)

    print("Read test/valid data: method-summary pair %d" % (len(summaries)))

    return summaries, methods_token, methods_sbt


if __name__ == "__main__":
    start = time.perf_counter()

    # main
    train_summaries, train_methods_token, train_methods_sbt, token_map_dict, sum_map_dict, sbt_map_dict, tok_word_count, \
    sum_word_count, sbt_word_count, tok_i2w, sum_i2w, sbt_i2w = read_train_data(
        file_path=config.train_data_path)
    val_summaries, val_methods_token, val_methods_sbt = read_test_data(file_path=config.val_data_path,
                                                                       tok_map_dict=token_map_dict,
                                                                       sum_map_dict=sum_map_dict,
                                                                       sbt_map_dict=sbt_map_dict)
    test_summaries, test_methods_token, test_methods_sbt = read_test_data(file_path=config.test_data_path,
                                                                          tok_map_dict=token_map_dict,
                                                                          sum_map_dict=sum_map_dict,
                                                                          sbt_map_dict=sbt_map_dict)
    # construct dataset dictionary
    dataset = {"train_summaries": train_summaries, "train_methods_token": train_methods_token,
               "train_methods_sbt": train_methods_sbt, "val_summaries": val_summaries,
               "val_methods_token": val_methods_token, "val_methods_sbt": val_methods_sbt,
               "test_summaries": test_summaries,
               "test_methods_token": test_methods_token, "test_methods_sbt": test_methods_sbt,
               "token_map_dict": token_map_dict, "sum_map_dict": sum_map_dict, "sbt_map_dict": sbt_map_dict,
               "tok_i2w": tok_i2w, "sum_i2w": sum_i2w, "sbt_i2w": sbt_i2w}

    # save all the data to pkl file
    output_dir = config.data_dir
    isExists = os.path.exists(output_dir)
    if not isExists:
        os.makedirs(output_dir)
    with open(output_dir + r'/dataset.pkl', 'wb') as output:
        pickle.dump(dataset, output)

    with open(output_dir + r'/tok_word_count.pkl', 'wb') as output:
        pickle.dump(tok_word_count, output)

    with open(output_dir + r'/sum_word_count.pkl', 'wb') as output:
        pickle.dump(sum_word_count, output)

    with open(output_dir + r'/sbt_word_count.pkl', 'wb') as output:
        pickle.dump(sbt_word_count, output)
    end = time.perf_counter()
    print("time cost: ", end - start)
