# calculate cliffs delta effect size: https://github.com/neilernst/cliffsDelta/blob/master/cliffsDelta.py
from __future__ import division
import copy
import random
from collections import Counter
import numpy as np
from prettytable import PrettyTable

def cliffsDelta(lst1, lst2, **dull):
    """Returns delta and true if there are more than 'dull' differences"""
    if not dull:
        dull = {'small': 0.147, 'medium': 0.33, 'large': 0.474} # effect sizes from (Hess and Kromrey, 2004)
    m, n = len(lst1), len(lst2)
    lst2 = sorted(lst2)
    j = more = less = 0
    for repeats, x in runs(sorted(lst1)):
        while j <= (n - 1) and lst2[j] < x:
            j += 1
        more += j*repeats
        while j <= (n - 1) and lst2[j] == x:
            j += 1
        less += (n - j)*repeats
    d = (more - less) / (m*n)
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
    statistic,pvalue = stats.wilcoxon(y1, y2)
    return pvalue


def get_score(y1,y2):
    pvalue = wilcoxon_signed_rank_test(y1, y2)
    d, size = cliffsDelta(y1, y2)
    return  pvalue, d, size


def get_pvalue_and_effect_size(all_score):
    models_name = list(all_score)
    for i in range(len( models_name )):
        for j in range(i+1,len( models_name )):
            pvalue, d, size = get_score(all_score[models_name[i]],all_score[models_name[j]])
            print("{} and {}, pvalue:{}, cliffsDelta:{}, effect size:{}".format(models_name[i],models_name[j],pvalue, d, size))


def get_all_model_score(path,question_cnt = 50):
    data_frame=pd.read_excel(path)
    result = {}
    
    user_cnt = len(data_frame["ID"])
    for i in range(user_cnt):
        for q in range(question_cnt):
            cocogum , ast_att_gru, astnn, rencos = list(data_frame.loc[i])[5+4*q:9+4*q]
            key = "Q_" + str(q)
            if key in result:
                result[ key][ "cocogum" ].append(cocogum)
                result[ key][ "ast_att_gru" ].append( ast_att_gru)
                result[ key][ "astnn" ].append(astnn)
                result[ key][ "rencos" ].append(rencos)
            else:
                result[ key] = {}
                result[ key][ "cocogum" ] = [cocogum]
                result[ key][ "ast_att_gru" ] = [ast_att_gru]
                result[ key][ "astnn" ] = [astnn]
                result[ key][ "rencos" ] = [rencos]

    cocogum_scores = []
    ast_att_gru_scores = []
    astnn_scores = []
    rencos_scores = []
    for q, four_score in result.items():
        cocogum_scores.extend(four_score["cocogum"])
        ast_att_gru_scores.extend(four_score["ast_att_gru"])
        astnn_scores.extend(four_score["astnn"])
        rencos_scores.extend(four_score["rencos"])

    all_score = {"cocogum":cocogum_scores,"ast_att_gru":ast_att_gru_scores,"astnn":astnn_scores,"rencos":rencos_scores }
    return all_score


def parse_score_dict(result):
    cocogum_scores = []
    ast_att_gru_scores = []
    astnn_scores = []
    rencos_scores = []
    for q, four_score in result.items():
        cocogum_scores.extend(four_score["cocogum"])
        ast_att_gru_scores.extend(four_score["ast_att_gru"])
        astnn_scores.extend(four_score["astnn"])
        rencos_scores.extend(four_score["rencos"])

    all_score = {"cocogum":cocogum_scores,"ast_att_gru":ast_att_gru_scores,"astnn":astnn_scores,"rencos":rencos_scores }
    return all_score


def get_all_model_in_three_aspects_score(path,question_cnt = 10,start_qid=1):
    model_order_dict = {}
    for i in range(1, 51,1):
        random.seed(i)
        li= [1,2,3,4]
        random.shuffle(li)
        model_order_dict[i] =  li    
#     print('model_order_dict:', model_order_dict)
    
#     path = "112506357_2_Code Summarization Human Evaluation 1- 10_2_2.xlsx"
    data_frame=pd.read_excel(path)
    result = {"informative":{}, "naturalness":{}, "similarity":{}}
    user_cnt = len(data_frame["序号"])
    for i in range(user_cnt):
        for q in range(question_cnt):
            start_index = 6 + q*12
            one_question_score = [list(data_frame.loc[0])[start_index+j*3:start_index+(j+1)*3] for j in range(4)]
            model_order_in_this_question = model_order_dict[q+start_qid]
            one_question_model_socre = dict (zip(model_order_in_this_question,one_question_score  ))
            ast_att_gru, astnn, rencos, cocogum = one_question_model_socre[1], \
                                                  one_question_model_socre[2], \
                                                  one_question_model_socre[3], \
                                                  one_question_model_socre[4]
            key = "Q_" + str(q)
            if key in result["informative"]:
                result["informative"][ key][ "cocogum" ].append(cocogum[0]-1)
                result["informative"][ key][ "ast_att_gru" ].append( ast_att_gru[0]-1)
                result["informative"][ key][ "astnn" ].append(astnn[0]-1)
                result["informative"][ key][ "rencos" ].append(rencos[0]-1)

                result["naturalness"][ key][ "cocogum" ].append(cocogum[1]-1)
                result["naturalness"][ key][ "ast_att_gru" ].append( ast_att_gru[1]-1)
                result["naturalness"][ key][ "astnn" ].append(astnn[1]-1)
                result["naturalness"][ key][ "rencos" ].append(rencos[1]-1)

                result["similarity"][ key][ "cocogum" ].append(cocogum[2]-1)
                result["similarity"][ key][ "ast_att_gru" ].append( ast_att_gru[2]-1)
                result["similarity"][ key][ "astnn" ].append(astnn[2]-1)
                result["similarity"][ key][ "rencos" ].append(rencos[2]-1)
            else:
                result["informative"] [ key] = {}
                result["naturalness"][ key]= {}
                result["similarity"][ key] = {}

                result["informative"][ key][ "cocogum" ]= [cocogum[0]-1]
                result["informative"][ key][ "ast_att_gru" ]=  [ast_att_gru[0]-1]
                result["informative"][ key][ "astnn" ] = [astnn[0]-1]
                result["informative"][ key][ "rencos" ] = [rencos[0]-1]

                result["naturalness"][ key][ "cocogum" ] = [cocogum[1]-1]
                result["naturalness"][ key][ "ast_att_gru" ] = [ast_att_gru[1]-1-1]
                result["naturalness"][ key][ "astnn" ] = [astnn[1]-1]
                result["naturalness"][ key][ "rencos" ] = [rencos[1]-1]

                result["similarity"][ key][ "cocogum" ] = [cocogum[2]-1]
                result["similarity"][ key][ "ast_att_gru" ] =  [ast_att_gru[2]-1]
                result["similarity"][ key][ "astnn" ] = [astnn[2]-1]
                result["similarity"][ key][ "rencos" ] = [rencos[2]-1]
    return  parse_score_dict(result['informative']), parse_score_dict(result['naturalness']), parse_score_dict(result['similarity'])


def print_distribution(four_model_score):
    table = PrettyTable(['model type', "0", "1", "2", "3", "4", "Avg(Std)", "≥3", "≥2", "≤1"])
    for k in four_model_score:
        result = Counter(four_model_score[k])
        avg = np.mean(four_model_score[k])
        std = np.std(four_model_score[k])
        table.add_row([k, result[0], result[1], result[2], result[3], result[4],
                       "{}({})".format(round(avg,2), round(std,2)),
                       result[3]+result[4], result[2]+result[3]+result[4], result[0]+result[1]])
    print(table)


# multi-excel
def merge_all_score(s1,s5):
    merged_scores = copy.deepcopy(s1)
    five_score = [s1,s5 ]
    for key in merged_scores[0].keys():
        for i in range(3):
            for j in range(1,len(five_score)):
                merged_scores[i][key].extend(five_score[j][i][key])
    return  merged_scores


def calcute_final_result(path1,path5):
    all_scores1_10 = get_all_model_in_three_aspects_score(path1,question_cnt=10,start_qid=1)

    all_scores41_50 = get_all_model_in_three_aspects_score(path5,question_cnt=10,start_qid=41)
    merged_scores = merge_all_score(all_scores1_10,all_scores41_50 )
    # print("informative")
    # print_distribution( merged_scores[0])
    # get_pvalue_and_effect_size( merged_scores[0])
    
    print(80*"*")
    print("naturalness")
    print_distribution( merged_scores[1])
    get_pvalue_and_effect_size( merged_scores[1])
    
    # print(80*"*")
    # print("similarity")
    # print_distribution( merged_scores[2])
    # get_pvalue_and_effect_size( merged_scores[2])
    return merged_scores

import krippendorff
def get_alpha_for_2_raters_in_one_group_question_one_approach_concat_2_aspect(all_scores, approach):
    rater0_in_three_aspects = []
    rater1_in_three_aspects = []
    rater2_in_three_aspects = []
    rater3_in_three_aspects = []

    aspects = {0: "informative", 1: "naturalness", }
    # for aspect in range(2):
    aspect=1
    score_cocogum = all_scores[aspect][approach]
    # rater0_range = list(range(0, 20, 2))
    # rater1_range = list(range(1, 20, 2))
    rater0_range = list(range(0, 40, 4))
    rater1_range = list(range(1, 40, 4))
    rater2_range = list(range(1, 40, 4))
    rater3_range = list(range(1, 40, 4))
    rater0 = [score_cocogum[item] for item in rater0_range]
    rater1 = [score_cocogum[item] for item in rater1_range]
    rater2 = [score_cocogum[item] for item in rater2_range]
    rater3 = [score_cocogum[item] for item in rater3_range]
    rater0_in_three_aspects.extend(rater0)
    rater1_in_three_aspects.extend(rater1)
    rater2_in_three_aspects.extend(rater2)
    rater3_in_three_aspects.extend(rater3)
    all_raters_score = [rater0_in_three_aspects, rater1_in_three_aspects, rater2_in_three_aspects, rater3_in_three_aspects]
    print(" %s in naturalness: " % (approach) + "Krippendorff's alpha for ordinal metric: {}".format(
        krippendorff.alpha(reliability_data=all_raters_score, level_of_measurement='ordinal')))


def main():
    path1_10 = r"112506357_2_Code Summarization Human Evaluation 1- 10_4_4.xlsx"
    # path1_10 = r"112504929_2_Code Summarization Human Evaluation 41- 50_4_4.xlsx"
    path41_50 = r"112506604_2_Code Summarization Human Evaluation 31- 40_4_4.xlsx"
    scores = calcute_final_result(path1_10, path41_50)
    approaches = ['cocogum', 'ast_att_gru', 'astnn', 'rencos']
    for approach in approaches:
        get_alpha_for_2_raters_in_one_group_question_one_approach_concat_2_aspect(scores, approach)

if __name__ == "__main__":
    """ run the following command:
    python human_eval.py 2>&1 | tee human_eval_result.log
    """
    main()