# -*- coding: utf-8 -*-
# @Time      : 2020-03-30 17:01
# @Author    : Eason Hu
# @Site      : 
# @File      : build_test_tree_depth.py

import torch
from lib.data.Tree import *


def get_data_trees(trees):
    data_trees = []
    for t_json in trees:
        for k, node in t_json.items():
            # if 'parent' not in node:
            #     root_idx ='#'
            #     break
            if node['parent'] == None and node['node'] != 'StatementExpression':
                root_idx = k
        # if root_idx == "#":
        #     continue
        tree = json2tree_binary(t_json, Tree(), root_idx)
        data_trees.append(tree)

    return data_trees


def bulid_test_index(num1, num2, num3, binary_tree):
    count = 0
    test1 = []
    test2 = []
    for i, t in enumerate(binary_tree):
        if count >= 138:
            break
        if t.leaf_count() < num1 and len(test2) < 128:
            test2.append(i)
            count += 1
        elif num2 < t.leaf_count() < num3 and len(test1) < 10:
            test1.append(i)
            count += 1
    test_data1_index = test2[:118] + test1
    test_data2_index = test2
    return test_data1_index, test_data2_index


def build_test_data(data_index, data):
    test_data = {'train_xe': {}}
    src = []
    tgt = []
    trees = []
    for idx in data_index:
        src.append(data['train_xe']['src'][idx])
        tgt.append(data['train_xe']['tgt'][idx])
        trees.append(data['train_xe']['trees'][idx])
    test_data['train_xe']['src'] = src
    test_data['train_xe']['tgt'] = tgt
    test_data['train_xe']['trees'] = trees
    test_data['test'] = test_data['train_xe']
    test_data['train_pg'] = test_data['train_xe']
    test_data['valid'] = test_data['train_xe']
    test_data['dicts'] = data['dicts']
    return test_data


def main(num1, num2, num3, data, binary_tree):
    save_path = '/mnt/yuxuan/Code_summarization0330/csn_dataset/test_diff_tree_depth/'
    test_data1_index, test_data2_index = bulid_test_index(num1, num2, num3, binary_tree)
    test_data1 = build_test_data(test_data1_index, data)
    test_data2 = build_test_data(test_data2_index, data)
    print('save_data to:', save_path)
    torch.save(test_data1, save_path + '118_lt_{}_10_rt_{}_lt_{}.pt'.format(num1, num2, num3))
    torch.save(test_data2, save_path + '128_lt_{}.pt'.format(num1))

if __name__ == '__main__':
    print('start')
    data_path = '/mnt/yuxuan/Code_summarization0330/csn_dataset/train_data0330/processed_filter.train.pt2'
    data = torch.load(data_path)
    binary_tree = get_data_trees(data['train_xe']['trees'])
    test1 = [30, 2000, 10000]
    test2 = [30, 30, 47]
    test3 = [47, 47, 70]
    test = [test1, test2, test3]
    for t in test:
        num1, num2, num3 = t
        main(num1, num2, num3, data, binary_tree)