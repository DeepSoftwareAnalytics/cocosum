#!/usr/bin/env python
#!-*-coding:utf-8 -*- 
'''
@version: python3.7
@author: ‘enshi‘
@license: Apache Licence 
@contact: *******@qq.com
@site: 
@software: PyCharm
@file: data_of _leclair.py
@time: 1/20/2020 1:56 PM
'''

import pickle
from util.Config import Config as cf
import os



def read_leclair(seqdata,tt = 'train',):
    allfids = list(seqdata['d%s' % (tt)].keys())
    tokseqs = list()
    sumseqs = list()
    sbtseqs = list()
    for fid in allfids:
        wtokseq = seqdata['d%s' % (tt)][fid].tolist()
        wsumseq = seqdata['c%s' % (tt)][fid].tolist()[1:]
        wsbtseq = seqdata['s%s' % (tt)][fid].tolist()
        tokseqs.append(wtokseq)
        sumseqs.append(wsumseq )
        sbtseqs.append(wsbtseq)
    return sumseqs, tokseqs,sbtseqs

#please use your path
# dat_path = r"./funcom/data/standard/dataset.pkl"
dat_path = r"/datadrive/CodeSum/data/funcom/data/standard/dataset.pkl"
seqdata = pickle.load(open(dat_path , 'rb'))

#like before
train_summaries, train_methods_token, train_methods_sbt = read_leclair(seqdata,tt = 'train',)
val_summaries, val_methods_token, val_methods_sbt= read_leclair(seqdata,tt = 'val',)
test_summaries, test_methods_token, test_methods_sbt = read_leclair(seqdata,tt = 'test',)

token_map_dict = seqdata["datstok"].w2i
sbt_map_dict = seqdata["smltok"].w2i
sum_map_dict = seqdata["comstok"].w2i

# There has error about sbt_i2w and sum_i2w
tok_i2w = seqdata["datstok"].i2w
sbt_i2w = seqdata["smltok"].i2w
sum_i2w = seqdata["comstok"].i2w


# construct dataset dictionary
dataset = {"train_summaries": train_summaries, "train_methods_token": train_methods_token,
           "train_methods_sbt": train_methods_sbt, "val_summaries": val_summaries,
           "val_methods_token": val_methods_token, "val_methods_sbt": val_methods_sbt, "test_summaries": test_summaries,
           "test_methods_token": test_methods_token, "test_methods_sbt": test_methods_sbt,

           "token_map_dict": token_map_dict, "sum_map_dict": sum_map_dict, "sbt_map_dict": sbt_map_dict,
           "tok_i2w": tok_i2w, "sum_i2w": sum_i2w, "sbt_i2w": sbt_i2w}

# save all the data to pkl file
output_dir = cf.data_dir
isExists = os.path.exists(output_dir)
if not isExists:
    os.makedirs(output_dir)
with open(output_dir + r'/datase  .pkl', 'wb') as output:
    pickle.dump(dataset, output)