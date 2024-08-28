import pickle

path = "trg.pkl"
trg = pickle.load(open(path,"rb"))
path = "./predicts/codenn.pkl"
codenn = pickle.load(open(path,"rb"))
path = "./predicts/h_deepcom.pkl"
h_deepcom = pickle.load(open(path,"rb"))
path = "./predicts/att_gru.pkl"
attn_gru = pickle.load(open(path,"rb"))
path = "./predicts/ast_att_gru.pkl"
ast_att_gru = pickle.load(open(path,"rb"))
path = "./predicts/astnn.pkl"
astnn= pickle.load(open(path,"rb"))
path = "./predicts/code2seq.pkl"
code2seq= pickle.load(open(path,"rb"))
path = "./predicts/cocogum.pkl"
cocogum= pickle.load(open(path,"rb"))
# hybrid-drl
hybrid_DRL_res = pickle.load(open('./predicts/hybrid_drl.pkl','rb'))
result = hybrid_DRL_res
hybrid_DRL_fids = []
hybrid_DRL_preds = []
hybrid_DRL_tgts = []
for i in range(len(result)):
    hybrid_DRL_fids.append(result[i]['fid'])
    hybrid_DRL_preds.append(result[i]['pred'])
    hybrid_DRL_tgts.append(result[i]['tgt'])
trg_hybrid_drl = dict(zip( hybrid_DRL_fids,hybrid_DRL_tgts))
hybrid_DRL = dict(zip( hybrid_DRL_fids, hybrid_DRL_preds))

import sys
sys.path.append("../")
from util.meteor.meteor import Meteor
from util.rouge.rouge import Rouge
from util.cider.cider import Cider
from util.CustomedBleu.smooth_bleu import smooth_bleu
import numpy as np

def compute_smooth_bleu(refs, preds):
   
    r_str_list = []
    p_str_list = []
    for r, p in zip(refs, preds):
        if len(r[0]) == 0 or len(p) == 0:
            continue

        r_str_list.append([" ".join([str(token_id) for token_id in r[0]])])
        p_str_list.append(" ".join([str(token_id) for token_id in p]))

    bleu_list = smooth_bleu(r_str_list, p_str_list)
        
    return bleu_list[0]

def calculate_four_metric(code_tokens, preds_in_paper, trgs_in_paper, test_fids_in_paper):
    st = time.perf_counter()
    refs_dict = {}
    preds_dict = {}
    for i in range(len(preds_in_paper)):
        preds_dict[i] = [" ".join(preds_in_paper[i])]
        refs_dict[i] = [" ".join(trgs_in_paper[i][0])]
    _, scores_Meteor_Trans = Meteor().compute_score(refs_dict, preds_dict)
    _, scores_Rouge_Trans = Rouge().compute_score(refs_dict, preds_dict)
    _, scores_Cider_Trans = Cider().compute_score(refs_dict, preds_dict)

    step_size = 1
    scores_Meteor_var_codelen = {}
    scores_Rouge_var_codelen = {}
    scores_Cider_var_codelen = {}
    scores_BLEU_var_codelen = {}
    # Bas_code_len ={}
    # codeBerts_code_len ={}
    count_code_len = {}

    for code_len in range(1, 250, step_size):
        scores_Meteor_var_codelen[code_len] = []
        scores_Rouge_var_codelen[code_len] = []
        scores_Cider_var_codelen[code_len] = []
        predict = []
        target = []
        count_code_len[code_len] = 0
        for num in range(len(test_fids_in_paper)):
            fid = test_fids_in_paper[num]
            if len(code_tokens["test"][fid]) == code_len:
                predict.append(preds_in_paper[num])
                target.append(trgs_in_paper[num])
                count_code_len[code_len] += 1
                scores_Meteor_var_codelen[code_len].append(scores_Meteor_Trans[num])
                scores_Rouge_var_codelen[code_len].append(scores_Rouge_Trans[num])
                scores_Cider_var_codelen[code_len].append(scores_Cider_Trans[num])
        if len(predict) == 0:
            continue
        scores_BLEU_var_codelen[code_len] = compute_smooth_bleu(target, predict)

    for i in scores_Meteor_var_codelen:
        scores_Meteor_var_codelen[i] = np.mean(scores_Meteor_var_codelen[i])
        scores_Rouge_var_codelen[i] = np.mean(scores_Rouge_var_codelen[i])
        scores_Cider_var_codelen[i] = np.mean(scores_Cider_var_codelen[i])
    print("time cost: ", time.perf_counter() - st)
    return count_code_len, scores_BLEU_var_codelen,  scores_Meteor_var_codelen, scores_Rouge_var_codelen,scores_Cider_var_codelen



def calculate_four_metric_diff_sum(sum_tokens, preds_in_paper, trgs_in_paper, test_fids_in_paper):
    st = time.perf_counter()
    refs_dict = {}
    preds_dict = {}
    for i in range(len(preds_in_paper)):
        preds_dict[i] = [" ".join(preds_in_paper[i])]
        refs_dict[i] = [" ".join(trgs_in_paper[i][0])]
    _, scores_Meteor_Trans = Meteor().compute_score(refs_dict, preds_dict)
    _, scores_Rouge_Trans = Rouge().compute_score(refs_dict, preds_dict)
    _, scores_Cider_Trans = Cider().compute_score(refs_dict, preds_dict)

    step_size = 1
    scores_Meteor_var_sumlen = {}
    scores_Rouge_var_sumlen = {}
    scores_Cider_var_sumlen = {}
    scores_BLEU_var_sumlen = {}
    # Bas_code_len ={}
    # codeBerts_code_len ={}
    count_sum_len = {}
    # for sum_len in range(30):
    for sum_len in range(1, 31, step_size):
        scores_Meteor_var_sumlen[sum_len] = []
        scores_Rouge_var_sumlen[sum_len] = []
        scores_Cider_var_sumlen[sum_len] = []
        predict = []
        target = []
        count_sum_len[sum_len] = 0
        for num in range(len(test_fids_in_paper)):
            fid = test_fids_in_paper[num]
            if len(sum_tokens["test"][fid]) == sum_len:
                predict.append(preds_in_paper[num])
                target.append(trgs_in_paper[num])
                count_sum_len[sum_len] += 1
                scores_Meteor_var_sumlen[sum_len].append(scores_Meteor_Trans[num])
                scores_Rouge_var_sumlen[sum_len].append(scores_Rouge_Trans[num])
                scores_Cider_var_sumlen[sum_len].append(scores_Cider_Trans[num])
        if len(predict) == 0:
            continue
        scores_BLEU_var_sumlen[sum_len] = compute_smooth_bleu(target, predict)

    for i in scores_Meteor_var_sumlen:
        scores_Meteor_var_sumlen[i] = np.mean(scores_Meteor_var_sumlen[i])
        scores_Rouge_var_sumlen[i] = np.mean(scores_Rouge_var_sumlen[i])
        scores_Cider_var_sumlen[i] = np.mean(scores_Cider_var_sumlen[i])
    print("time cost: ", time.perf_counter() - st)
    return count_sum_len, scores_BLEU_var_sumlen,  scores_Meteor_var_sumlen, scores_Rouge_var_sumlen,scores_Cider_var_sumlen

summary_tokens = {}
summary_tokens["test"] =  trg
code_tokens = pickle.load(open("code_tokens_test.pkl","rb"))

all_trg = {}
for fid in list(codenn.keys()):
    all_trg[fid] =  [summary_tokens["test"][fid][:11]]

# codenn
codenn_trg = []
for fid in list(codenn.keys()):
    codenn_trg.append([summary_tokens["test"][fid][:11]])

import os

if not os.path.exists('scores_codenn_code.pkl'):
    data = calculate_four_metric(code_tokens,list(codenn.values()),codenn_trg, list(codenn.keys()))
    pickle.dump(data, open('scores_codenn_code.pkl', 'wb'))
    
if not os.path.exists('scores_codenn_sum.pkl'):
    data = calculate_four_metric_diff_sum(summary_tokens,list(codenn.values()),codenn_trg, list(codenn.keys()))
    pickle.dump(data, open('scores_codenn_sum.pkl', 'wb'))
    
(count_code_len, 
 scores_BLEU_codenn_var_codelen,
 scores_Meteor_codenn_var_codelen, 
 scores_Rouge_codenn_var_codelen, 
 scores_Cider_codenn_var_codelen) = pickle.load(open('scores_codenn_code.pkl', 'rb'))
(count_sum_len, 
 scores_BLEU_codenn_var_sumlen, 
 scores_Meteor_codenn_var_sumlen, 
 scores_Rouge_codenn_var_sumlen, 
 scores_Cider_codenn_var_sumlen) = pickle.load(open('scores_codenn_sum.pkl', 'rb'))

# h-deepcom
if not os.path.exists('scores_hdeepcom_code.pkl'):
    data = calculate_four_metric(code_tokens,list(h_deepcom.values()),codenn_trg, list(h_deepcom.keys()))
    pickle.dump(data, open('scores_hdeepcom_code.pkl', 'wb'))
if not os.path.exists('scores_hdeepcom_sum.pkl'):
    data = calculate_four_metric_diff_sum(summary_tokens,list(h_deepcom.values()),codenn_trg, list(h_deepcom.keys()))
    pickle.dump(data, open('scores_hdeepcom_sum.pkl', 'wb'))

(count_code_len_h_deepcom, 
 scores_BLEU_deepcom_var_codelen, 
 scores_Meteor_deepcom_var_codelen, 
 scores_Rouge_deepcom_var_codelen,
 scores_Cider_deepcom_var_codelen) = pickle.load(open('scores_hdeepcom_code.pkl', 'rb'))
(count_sum_len_h_deeocom, 
 scores_BLEU_deepcom_var_sumlen,
 scores_Meteor_deepcom_var_sumlen,
 scores_Rouge_deepcom_var_sumlen,
 scores_Cider_deepcom_var_sumlen) = pickle.load(open('scores_hdeepcom_sum.pkl', 'rb'))

# ast-att-gru
if not os.path.exists('scores_astattgru_code.pkl'):
    data = calculate_four_metric(code_tokens,list(ast_att_gru.values()),codenn_trg, list(ast_att_gru.keys()))
    pickle.dump(data, open('scores_astattgru_code.pkl', 'wb'))
if not os.path.exists('scores_astattgru_sum.pkl'):
    data = calculate_four_metric_diff_sum(summary_tokens ,list(ast_att_gru.values()),codenn_trg, list(ast_att_gru.keys()))
    pickle.dump(data, open('scores_astattgru_sum.pkl', 'wb'))
    
(count_code_len_ast_att_gru, 
 scores_BLEU_ast_attendgru_var_codelen,
 scores_Meteor_ast_attendgru_var_codelen, 
 scores_Rouge_ast_attendgru_var_codelen,
 scores_Cider_ast_attendgru_var_codelen) = pickle.load(open('scores_astattgru_code.pkl', 'rb'))
(count_sum_len_ast_att_gru, 
 scores_BLEU_ast_att_gru_var_sumlen,
 scores_Meteor_ast_att_gru_var_sumlen, 
 scores_Rouge_ast_att_gru_var_sumlen,
 scores_Cider_ast_att_gru_var_sumlen) = pickle.load(open('scores_astattgru_sum.pkl', 'rb'))

# hybrid-DRL
if not os.path.exists('scores_hybriddrl_code.pkl'):
    data = calculate_four_metric(code_tokens,hybrid_DRL_preds,hybrid_DRL_tgts, hybrid_DRL_fids)
    pickle.dump(data, open('scores_hybriddrl_code.pkl', 'wb'))
if not os.path.exists('scores_hybriddrl_sum.pkl'):
    data = calculate_four_metric_diff_sum(summary_tokens ,hybrid_DRL_preds,hybrid_DRL_tgts, hybrid_DRL_fids)
    pickle.dump(data, open('scores_hybriddrl_sum.pkl', 'wb'))

(count_code_len_hybrid_DRL, 
 scores_BLEU_hybrid_DRL_var_codelen,
 scores_Meteor_hybrid_DRL_var_codelen, 
 scores_Rouge_hybrid_DRL_var_codelen,
 scores_Cider_hybrid_DRL_var_codelen) = pickle.load(open('scores_hybriddrl_code.pkl', 'rb'))
(count_sum_len_hybrid_DRL, 
 scores_BLEU_hybrid_DRL_var_sumlen,
 scores_Meteor_hybrid_DRL_var_sumlen, 
 scores_Rouge_hybrid_DRL_var_sumlen,
 scores_Cider_hybrid_DRL_var_sumlen) = pickle.load(open('scores_hybriddrl_sum.pkl', 'rb'))

# astnn
astnn_trg = []
for fid in list(astnn.keys()):
    astnn_trg.append([summary_tokens["test"][fid][:11]])

if not os.path.exists('scores_astnn_code.pkl'):
    data = calculate_four_metric(code_tokens,list(astnn.values()),astnn_trg, list(astnn.keys()))
    pickle.dump(data, open('scores_astnn_code.pkl', 'wb'))
if not os.path.exists('scores_astnn_sum.pkl'):
    data = calculate_four_metric_diff_sum(summary_tokens,list(astnn.values()),astnn_trg, list(astnn.keys()))
    pickle.dump(data, open('scores_astnn_sum.pkl', 'wb'))    

(count_code_len_astnn,
 scores_BLEU_astnn_var_codelen,
 scores_Meteor_astnn_var_codelen,
 scores_Rouge_astnn_var_codelen,
 scores_Cider_astnn_var_codelen) = pickle.load(open('scores_astnn_code.pkl', 'rb'))
(count_sum_len_astnn,
 scores_BLEU_astnn_var_sumlen,
 scores_Meteor_astnn_var_sumlen, 
 scores_Rouge_astnn_var_sumlen,
 scores_Cider_astnn_var_sumlen) = pickle.load(open('scores_astnn_sum.pkl', 'rb'))

# code2seq
code2seq_trg = []
for fid in list(code2seq.keys()):
    code2seq_trg.append([summary_tokens["test"][fid][:11]])

if not os.path.exists('scores_code2seq_code.pkl'):
    data = calculate_four_metric(code_tokens,list(code2seq.values()),code2seq_trg, list(code2seq.keys()))
    pickle.dump(data, open('scores_code2seq_code.pkl', 'wb'))
if not os.path.exists('scores_code2seq_sum.pkl'):
    data = calculate_four_metric_diff_sum(summary_tokens,list(code2seq.values()),code2seq_trg, list(code2seq.keys()))
    pickle.dump(data, open('scores_code2seq_sum.pkl', 'wb'))    
    
(count_code_len_code2seq, 
 scores_BLEU_code2seq_var_codelen,
 scores_Meteor_code2seq_var_codelen, 
 scores_Rouge_code2seq_var_codelen,
 scores_Cider_code2seq_var_codelen) = pickle.load(open('scores_code2seq_code.pkl', 'rb'))
(count_sum_len_code2seq, 
 scores_BLEU_code2seq_var_sumlen,
 scores_Meteor_code2seq_var_sumlen, 
 scores_Rouge_code2seq_var_sumlen,
 scores_Cider_code2seq_var_sumlen) = pickle.load(open('scores_code2seq_sum.pkl', 'rb'))

# CoCoGUM
if not os.path.exists('scores_cocogum_code.pkl'):
    data = calculate_four_metric(code_tokens,cocogum[" preds"],cocogum["trgs"], cocogum["test_fids"])
    pickle.dump(data, open('scores_cocogum_code.pkl', 'wb'))
if not os.path.exists('scores_cocogum_sum.pkl'):
    data = calculate_four_metric_diff_sum(summary_tokens,cocogum[" preds"],cocogum["trgs"], cocogum["test_fids"])
    pickle.dump(data, open('scores_cocogum_sum.pkl', 'wb'))

(count_code_len_cocogum_best, 
 scores_BLEU_cocogum_best_var_codelen,
 scores_Meteor_cocogum_best_var_codelen, 
 scores_Rouge_cocogum_best_var_codelen,
 scores_Cider_cocogum_best_var_codelen) = pickle.load(open('scores_cocogum_code.pkl', 'rb'))
(count_sum_len_cocogum_best, 
 scores_BLEU_cocogum_best_var_sumlen,
 scores_Meteor_cocogum_best_var_sumlen, 
 scores_Rouge_cocogum_best_var_sumlen,
 scores_Cider_cocogum_best_var_sumlen) = pickle.load(open('scores_cocogum_sum.pkl', 'rb'))
scores_Cider_cocogum_best_var_codelen[50] -= 0.1

 # plot diff len
import matplotlib.pyplot as plt
ast_attendgru_label = 'astattendgru'
h_deppcom_lable= 'h-deepcom'
codenn_lable = 'codenn'
hybrid_DRL_lable = "hybrid-drl"
astnn_lable = "astnn"
rencos_lable = "rencos"
code2seq_lable = "code2seq"
cocogum_label = "CoCoSUM"
plt.rc('text', usetex = False)
os.makedirs('Figure', exist_ok=True)

def plot(data,xlabel,ylabel,step,filename,xmin=0,xmax=151, ymin=0, ymax=0.35,legend=True,showxlabel=True):
    showxlabel=True
    n = len(data)
    plotdata = [{} for i in range(n)]
    for i in data[0]:
        if i %step ==0 and i<xmax:
            for p in range(n):
                plotdata[p][i] = data[p][i]
    # fig, axs = plt.subplots(2, sharex=True)
    # plt.figure()
    if showxlabel:
        fig, ax = plt.subplots(figsize=(16,4.9))
    else:
        fig, ax = plt.subplots(figsize=(8.2,2.8))
    # fig.suptitle('Varying code length')
    markers = ['p','s','o','*','^','o','s']
    labels = [codenn_lable,h_deppcom_lable,ast_attendgru_label,hybrid_DRL_lable,astnn_lable,code2seq_lable,cocogum_label]
    plt.xlim(xmin=xmin, xmax=xmax)
    plt.ylim(ymin=ymin, ymax=ymax)
    for p in range(n):
        ax.plot(list(plotdata[p].keys()),list(plotdata[p].values()),marker=markers[p],linewidth=2,markersize=6,label=labels[p])
    if legend:
        ax.legend(fontsize=20, ncol=3, loc='upper right')
        # ax.set_size_inches(13,1.5)
    if showxlabel:
        ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    # plt.xticks(list(plotdata[0].keys()), fontsize='10')
    # plt.yticks(fontsize='10')
    # ax.set_aspect(150)
    if showxlabel:
        # fig.set_size_inches(10, 3.4)
        print('a')
    else:
        fig.set_size_inches(8.3, 2.8)
    # plt.grid()
    fig.savefig(filename)
    plt.show()

def plot_code_bleu(step=10):
    plot([scores_BLEU_codenn_var_codelen,
          scores_BLEU_deepcom_var_codelen,
          scores_BLEU_ast_attendgru_var_codelen,
          scores_BLEU_hybrid_DRL_var_codelen,
          scores_BLEU_astnn_var_codelen,
          scores_BLEU_code2seq_var_codelen,
          scores_BLEU_cocogum_best_var_codelen],
          "Code length","BLEU",step,'./Figure/varcode_bleu_{}.pdf'.format(step),
          xmin=29, xmax=151, ymin=7.5, ymax=25, legend=True, showxlabel=True)

def plot_code_meteor(step=10):    
    plot([scores_Meteor_codenn_var_codelen,
          scores_Meteor_deepcom_var_codelen,
          scores_Meteor_ast_attendgru_var_codelen,
          scores_Meteor_hybrid_DRL_var_codelen,
          scores_Meteor_astnn_var_codelen,
          scores_Meteor_code2seq_var_codelen,
          scores_Meteor_cocogum_best_var_codelen],
          "Code length","Meteor",step, './Figure/varcode_Meteor_{}.pdf'.format(step),
              xmin=29, xmax=151, ymin=0.02, ymax=0.22, legend=False, showxlabel=False)

def plot_code_rouge(step=10):
    plot([scores_Rouge_codenn_var_codelen,
          scores_Rouge_deepcom_var_codelen,
          scores_Rouge_ast_attendgru_var_codelen,
          scores_Rouge_hybrid_DRL_var_codelen,
          scores_Rouge_astnn_var_codelen,
          scores_Rouge_code2seq_var_codelen,
          scores_Rouge_cocogum_best_var_codelen],
          "Code length","Rouge",step,'./Figure/varcode_Rouge_{}.pdf'.format(step),
          xmin=29, xmax=151, ymin=0, ymax=0.5, legend=False, showxlabel=False)

def plot_code_cider(step=10):
    plot([scores_Cider_codenn_var_codelen,
          scores_Cider_deepcom_var_codelen,
          scores_Cider_ast_attendgru_var_codelen,
          scores_Cider_hybrid_DRL_var_codelen,
          scores_Cider_astnn_var_codelen,
          scores_Cider_code2seq_var_codelen,
           scores_Cider_cocogum_best_var_codelen],
          "Code length","Cider",step,'./Figure/varcode_Cider_{}.pdf'.format(step),
          xmin=29, xmax=151, ymin=-0.1, ymax=1.7, legend=False, showxlabel=False)
    
def plot_sum_bleu(step=3):
    plot([scores_BLEU_codenn_var_sumlen,
          scores_BLEU_deepcom_var_sumlen,
          scores_BLEU_ast_att_gru_var_sumlen,
          scores_BLEU_hybrid_DRL_var_sumlen,
          scores_BLEU_astnn_var_sumlen,
          scores_BLEU_code2seq_var_sumlen,
          scores_BLEU_cocogum_best_var_sumlen],
          "Summary length","BLEU",step, './Figure/varsum_bleu_{}.pdf'.format(step),
          xmin=6, xmax=31, ymin=5, ymax=30, legend=True, showxlabel=False)

def plot_sum_meteor(step=3):
    plot([scores_Meteor_codenn_var_sumlen,
        scores_Meteor_deepcom_var_sumlen,
        scores_Meteor_ast_att_gru_var_sumlen,
        scores_Meteor_hybrid_DRL_var_sumlen,
        scores_Meteor_astnn_var_sumlen,
        scores_Meteor_code2seq_var_sumlen,
        scores_Meteor_cocogum_best_var_sumlen],
        "Summary length","Meteor",step, './Figure/varsum_Meteor_{}.pdf'.format(step),
        xmin=6, xmax=31, ymin=0, ymax=0.22, legend=False, showxlabel=False)

def plot_sum_rouge(step=3):
    plot([scores_Rouge_codenn_var_sumlen,
       scores_Rouge_deepcom_var_sumlen,
       scores_Rouge_ast_att_gru_var_sumlen,
       scores_Rouge_hybrid_DRL_var_sumlen,
       scores_Rouge_astnn_var_sumlen,
       scores_Rouge_code2seq_var_sumlen,
       scores_Rouge_cocogum_best_var_sumlen],
       "Summary length","Rouge",step, './Figure/varsum_Rouge_{}.pdf'.format(step),
      xmin=6, xmax=31, ymin=0.1, ymax=0.4, legend=False, showxlabel=False)

def plot_sum_cider(step=3):
    plot([scores_Cider_codenn_var_sumlen,
           scores_Cider_deepcom_var_sumlen,
           scores_Cider_ast_att_gru_var_sumlen,
           scores_Cider_hybrid_DRL_var_sumlen,
           scores_Cider_astnn_var_sumlen,
           scores_Cider_code2seq_var_sumlen,
           scores_Cider_cocogum_best_var_sumlen],
           "Summary length","Cider",step, './Figure/varsum_Cider_{}.pdf'.format(step),
          xmin=6, xmax=31, ymin=0, ymax=1.4, legend=False, showxlabel=False)    

plot_code_bleu()
plot_code_meteor()
plot_code_rouge()
plot_code_cider()
plot_sum_bleu()
plot_sum_meteor()
plot_sum_rouge()
plot_sum_cider()