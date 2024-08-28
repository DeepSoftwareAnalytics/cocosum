# -*- coding: utf-8 -*-
# @Time      : 2020-03-11 14:59
# @Author    : Eason Hu
# @Site      : 
# @File      : csn_vsm.py

import sys
sys.path.append('../..')
from spiral import ronin
import json
import re
from gensim import corpora, similarities, models
import pickle
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from util.EvaluateUtil import bleu_so_far

def build_dictionary(data):
    corpora_sen = []
    for idx in data.keys():
        corpora_sen.append(data[idx]['code_tokens_split'])
    dictionary = corpora.Dictionary(corpora_sen)
    return dictionary

def build_corpus(data, dictionary):
    corpus = []
    for idx in data.keys():
        temp = dictionary.doc2bow(data[idx]['code_tokens_split'])
        corpus.append(temp)
    return corpus

def build_tfidf_model(corpus):
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    similarity_tfidf = similarities.Similarity('./similarity_tfidf', corpus_tfidf, len(dictionary), num_best=5)

    return tfidf, similarity_tfidf

def eval_vsm(train_data, test_data, test_corpus, similarity_tfidf, tfidf):
    refs = []
    preds = []
    test_start_id = test_data.keys()[0]
    for i in range(len(test_corpus)):
        top_k_sim = similarity_tfidf[tfidf[test_corpus[i]]] #TODO this step is very slow, will add parallelize later
        pred_id = top_k_sim[0][0]
        pred = train_data[pred_id]['docstring_tokens_split']
        ref = test_data[test_start_id + i]['docstring_tokens_split']
        preds.append(pred)
        refs.append([ref])
    return bleu_so_far(refs, preds)

if __name__ == '__main__':

    train_data_dir = '/mnt/yuxuan/CodeSum/datapreprocess/CodeSearchNet/split_data/train/dataset_with_id_split_filter_all.pkl'
    train_data = pickle.load(open(train_data_dir, 'rb'))
    dictionary = build_dictionary(train_data)
    train_corpus = build_corpus(train_data, dictionary)

    test_data_dir = '/mnt/yuxuan/CodeSum/datapreprocess/CodeSearchNet/split_data/test/dataset_with_id_split_filter_all.pkl'
    test_data = pickle.load(open(test_data_dir, 'rb'))
    test_corpus = build_corpus(test_data, dictionary)

    tfidf, similarity_tfidf = build_tfidf_model(test_corpus)

    ret = eval_vsm(train_data, test_data, test_corpus, similarity_tfidf, tfidf)
    print(ret)