# -*- coding: utf-8 -*-
# @Time      : 2020-02-10 11:11
# @Author    : Eason Hu
# @Site      : 
# @File      : change_dataset.py

from util.DataUtil import read_pickle_data
from util.Config import Config as cf
import pickle
import os

"""
This is file to change leclair dataset.pkl to our token.
"""

def build_vocab(i2w, new_i2w, origin_dict, new_dict):
    i2w[0] = '<Padding>'
    origin_dict['<Padding>'] = 0
    for k in origin_dict.keys():
        if k in new_dict:
            continue
        new_dict[k] = len(new_dict)
        new_i2w[new_dict[k]] = k
    return new_dict, new_i2w

def change_sequences(new_dict, i2w, index_sequences):
    for seq in index_sequences:
        for i in range(len(seq)):
            if seq[i] == 0:
                break
            seq[i] = new_dict[i2w[seq[i]]]
    return index_sequences

def change(data):
    train_summaries = data["train_summaries"]
    train_methods_token = data["train_methods_token"]
    train_methods_sbt = data["train_methods_sbt"]

    # vocabulary info
    token_map_dict = data["token_map_dict"]
    sum_map_dict = data["sum_map_dict"]
    sbt_map_dict = data["sbt_map_dict"]

    # id2word info
    tok_i2w = data["tok_i2w"]
    sum_i2w = data["sbt_i2w"]
    sbt_i2w = data["sum_i2w"]
    # sbt_i2w = data["sbt_i2w"]

    sum_map_dict["<SOS>"] = cf.SOS_token_id
    sum_map_dict["<EOS>"] = cf.EOS_token_id
    sum_i2w[1] = "<SOS>"
    sum_i2w[2] = "<EOS>"
    # val info
    val_summaries = data["val_summaries"]
    val_methods_token = data["val_methods_token"]
    val_methods_sbt = data["val_methods_sbt"]
    # test info
    test_summaries = data["test_summaries"]
    test_methods_token = data["test_methods_token"]
    test_methods_sbt = data["test_methods_sbt"]

    new_token_dict = {"<Padding>": cf.PAD_token_id, "<SOS>": cf.SOS_token_id, "<EOS>": cf.EOS_token_id,
                    "<UNK>": cf.UNK_token_id}
    new_sum_dict = {"<Padding>": cf.PAD_token_id, "<SOS>": cf.SOS_token_id, "<EOS>": cf.EOS_token_id,
                    "<UNK>": cf.UNK_token_id}
    new_sbt_dict = {"<Padding>": cf.PAD_token_id, "<SOS>": cf.SOS_token_id, "<EOS>": cf.EOS_token_id,
                    "<UNK>": cf.UNK_token_id}

    new_tok_i2w = {cf.PAD_token_id: "<Padding>", cf.SOS_token_id: "<SOS>", cf.EOS_token_id: "<EOS>",
                   cf.UNK_token_id: "<UNK>"}

    new_sum_i2w = {cf.PAD_token_id: "<Padding>", cf.SOS_token_id: "<SOS>", cf.EOS_token_id: "<EOS>",
                   cf.UNK_token_id: "<UNK>"}

    new_sbt_i2w = {cf.PAD_token_id: "<Padding>", cf.SOS_token_id: "<SOS>", cf.EOS_token_id: "<EOS>",
                   cf.UNK_token_id: "<UNK>"}

    new_token_dict, new_tok_i2w = build_vocab(tok_i2w, new_tok_i2w, token_map_dict, new_token_dict)
    new_sum_dict, new_sum_i2w = build_vocab(sum_i2w, new_sum_i2w, sum_map_dict, new_sum_dict)
    new_sbt_dict, new_sbt_i2w = build_vocab(sbt_i2w, new_sbt_i2w, sbt_map_dict, new_sbt_dict)

    # new train info
    train_summaries = change_sequences(new_sum_dict, sum_i2w, train_summaries)
    train_methods_token = change_sequences(new_token_dict, tok_i2w, train_methods_token)
    train_methods_sbt = change_sequences(new_sbt_dict, sbt_i2w, train_methods_sbt)

    # new val info
    val_summaries = change_sequences(new_sum_dict, sum_i2w, val_summaries)
    val_methods_token = change_sequences(new_token_dict, tok_i2w, val_methods_token)
    val_methods_sbt = change_sequences(new_sbt_dict, sbt_i2w, val_methods_sbt)

    # new test info
    test_summaries = change_sequences(new_sum_dict, sum_i2w, test_summaries)
    test_methods_token = change_sequences(new_token_dict, tok_i2w, test_methods_token)
    test_methods_sbt = change_sequences(new_sbt_dict, sbt_i2w, test_methods_sbt)

    return train_summaries, train_methods_token, train_methods_sbt, val_summaries, val_methods_token, val_methods_sbt, test_summaries, test_methods_token, test_methods_sbt, new_token_dict, new_sum_dict, new_sbt_dict, new_tok_i2w, new_sum_i2w, new_sbt_i2w

if __name__ == '__main__':
    data = read_pickle_data(cf.in_path)
    train_summaries, train_methods_token, train_methods_sbt, val_summaries, val_methods_token, val_methods_sbt, \
    test_summaries, test_methods_token, test_methods_sbt, token_map_dict, sum_map_dict, sbt_map_dict, \
    tok_i2w, sum_i2w, sbt_i2w = change(data)

    dataset = {"train_summaries": train_summaries, "train_methods_token": train_methods_token,
               "train_methods_sbt": train_methods_sbt, "val_summaries": val_summaries,
               "val_methods_token": val_methods_token, "val_methods_sbt": val_methods_sbt,
               "test_summaries": test_summaries,
               "test_methods_token": test_methods_token, "test_methods_sbt": test_methods_sbt,
               "token_map_dict": token_map_dict, "sum_map_dict": sum_map_dict, "sbt_map_dict": sbt_map_dict,
               "tok_i2w": tok_i2w, "sum_i2w": sum_i2w, "sbt_i2w": sbt_i2w}

    output_dir = cf.data_dir
    isExists = os.path.exists(output_dir)
    if not isExists:
        os.makedirs(output_dir)
    with open(output_dir + r'/dataset_change.pkl', 'wb') as output:
        pickle.dump(dataset, output)