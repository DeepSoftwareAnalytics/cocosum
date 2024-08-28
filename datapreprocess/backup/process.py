# -*- coding: utf-8 -*-
# @Time      : 2020-02-05 14:47
# @Author    : Eason Hu
# @Site      : 
# @File      : process.py

from datapreprocess import get_ast
from datapreprocess.CodeSearchNet import code_search_net
import json

def load_data():
    tok_path = 'F:/msra/dataset/funcom_tokenized/functions'
    sum_path = 'F:/msra/dataset/funcom_tokenized/comments'
    sbt_path = 'F:/msra/dataset/funcom_processed/functions.json'

    with open(tok_path, 'r') as tok_fp:
        toks = {}
        lines = tok_fp.readlines()
        for line in lines:
            tok = line.split('\t')
            toks[tok[0]] = tok[1]

    with open(sum_path, 'r') as sum_fp:
        sums = {}
        lines = sum_fp.readlines()
        for line in lines:
            sum = line.split('\t')
            sums[sum[0]] = sum[1]
    error = 0
    with open(sbt_path, 'r') as sbt_fp:
        sbts = {}
        sbt = json.load(sbt_fp)
        for k, v in sbt.items():
            try:
                tokens = get_ast.process_source(v)
                code = " ".join(tokens)
                outputs = get_ast.get_ast(code)
                root = outputs[0]
                v = " ".join(get_ast.SBT(root, outputs))
                sbts[k] = v
            except:
                error += 1
    with open('sbt_data.json', 'w') as f:
        json.dump(sbts, f)
    print(error)
    print(len(sbts))


if __name__ == '__main__':

    load_data()