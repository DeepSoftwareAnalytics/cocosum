#!/usr/bin/env python
# !-*-coding:utf-8 -*-


### calculate cliffs delta effect size
### From https://github.com/neilernst/cliffsDelta/blob/master/cliffsDelta.py
from __future__ import division


def cliffsDelta(lst1, lst2, **dull):
    """Returns delta and true if there are more than 'dull' differences"""
    if not dull:
        dull = {'small': 0.147, 'medium': 0.33, 'large': 0.474}  # effect sizes from (Hess and Kromrey, 2004)
    m, n = len(lst1), len(lst2)
    lst2 = sorted(lst2)
    j = more = less = 0
    for repeats, x in runs(sorted(lst1)):
        while j <= (n - 1) and lst2[j] < x:
            j += 1
        more += j * repeats
        while j <= (n - 1) and lst2[j] == x:
            j += 1
        less += (n - j) * repeats
    d = (more - less) / (m * n)
    size = lookup_size(d, dull)
    return d, size


def lookup_size(delta: float, dull: dict) -> str:
    """
    :type delta: float
    :type dull: dict, a dictionary of small, medium, large thresholds.
    """
    delta = abs(delta)
    if delta < dull['small']:
        return 'negligible'
    if dull['small'] <= delta < dull['medium']:
        return 'small'
    if dull['medium'] <= delta < dull['large']:
        return 'medium'
    if delta >= dull['large']:
        return 'large'


def runs(lst):
    """Iterator, chunks repeated values"""
    for j, two in enumerate(lst):
        if j == 0:
            one, i = two, 0
        if one != two:
            yield j - i, one
            i = j
        one = two
    yield j - i + 1, two


### calculate wilcoxon pvalue
import pandas as pd
from scipy import stats


def wilcoxon_signed_rank_test(y1, y2):
    statistic, pvalue = stats.wilcoxon(y1, y2)
    return pvalue


def get_score(y1, y2):
    pvalue = wilcoxon_signed_rank_test(y1, y2)
    d, size = cliffsDelta(y1, y2)
    return pvalue, d, size


def get_pvalue_and_effect_size(all_score):
    models_name = list(all_score)
    for i in range(len(models_name)):
        for j in range(i + 1, len(models_name)):
            pvalue, d, size = get_score(all_score[models_name[i]], all_score[models_name[j]])
            print(
                "{} and {}, pvalue:{}, cliffsDelta:{}, effect size:{}".format(models_name[i], models_name[j], pvalue, d,
                                                                              size))


def get_all_model_score(path, question_cnt=50):
    data_frame = pd.read_excel(path)
    result = {}

    user_cnt = len(data_frame["ID"])
    for i in range(user_cnt):
        for q in range(question_cnt):
            cocogum, ast_att_gru, astnn, rencos = list(data_frame.loc[i])[5 + 4 * q:9 + 4 * q]
            key = "Q_" + str(q)
            if key in result:
                result[key]["cocogum"].append(cocogum)
                result[key]["ast_att_gru"].append(ast_att_gru)
                result[key]["astnn"].append(astnn)
                result[key]["rencos"].append(rencos)
            else:
                result[key] = {}
                result[key]["cocogum"] = [cocogum]
                result[key]["ast_att_gru"] = [ast_att_gru]
                result[key]["astnn"] = [astnn]
                result[key]["rencos"] = [rencos]

    cocogum_scores = []
    ast_att_gru_scores = []
    astnn_scores = []
    rencos_scores = []
    for q, four_score in result.items():
        cocogum_scores.extend(four_score["cocogum"])
        ast_att_gru_scores.extend(four_score["ast_att_gru"])
        astnn_scores.extend(four_score["astnn"])
        rencos_scores.extend(four_score["rencos"])

    all_score = {"cocogum": cocogum_scores, "ast_att_gru": ast_att_gru_scores, "astnn": astnn_scores,
                 "rencos": rencos_scores}
    return all_score


from collections import Counter
import numpy as np


def print_distribution(four_model_score):
    print('-' * 90)
    print("model type \t", "1\t", " 2\t", "3\t", "4\t", "5\t", "Avg\t", "≥4\t", "≥3\t", "≤2\t")
    for k in four_model_score:
        result = Counter(four_model_score[k])
        avg = np.mean(four_model_score[k])
        std = np.std(four_model_score[k])
        print(k, "  \t", result[1], "\t", result[2], "\t", result[3], "\t", result[4], "\t", result[5], "\t",
              "%.2f(%.2f)"%(avg,std), "\t", \
              result[4] + result[5], "\t", result[3] + result[4] + result[5], "\t", result[1] + result[2], "\t")
    print('-' * 90)


def print_all(path, question_cnt=50):
    all_score = get_all_model_score(path, question_cnt=50)
    print_distribution(all_score)
    get_pvalue_and_effect_size(all_score)

print_all(r"Source code summarization human evaluation（random50)(new)(1-20).xlsx", question_cnt=50)
