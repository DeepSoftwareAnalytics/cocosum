import pickle
import os, sys
sys.path.append("../../")
from util.DataUtil import open_json


# code =  pickle.load(open('/mnt/xiaodi/cocogum_refactor/csn_mini_data/code/djl1_dfp0_dsi1_dlc1_dr1.pkl', 'rb'))
code =  pickle.load(open('/mnt/xiaodi/cocogum_refactor/csn_mini_data/code/djl1_dfp0_dsi1_dlc0_dr1.pkl', 'rb'))
code0 = pickle.load(open('/mnt/xiaodi/cocogum_refactor/csn_mini_data/code/djl1_dfp0_dsi1_dlc1_dr1.pkl0', 'rb'))
print('code unchanged: ', code == code0)

summary = pickle.load(open('/mnt/xiaodi/cocogum_refactor/csn_mini_data/summary/cfp1_csi1_cfd0_clc1.pkl', 'rb'))
summary0 = pickle.load(open('/mnt/xiaodi/cocogum_refactor/csn_mini_data/summary/cfp1_csi1_cfd0_clc1.pkl0', 'rb'))
print('summary unchanged: ', summary == summary0)

sbt = pickle.load(open('/mnt/xiaodi/cocogum_refactor/csn_mini_data/sbt/sbts.pkl', 'rb'))
sbt0 = pickle.load(open('/mnt/xiaodi/cocogum_refactor/csn_mini_data/sbt/sbts.pkl0', 'rb'))
print('sbt unchanged: ', sbt == sbt0)

print('java file unchanged: ')
os.system("diff /mnt/xiaodi/cocogum_refactor/csn_mini_data/java_files/454451.java /mnt/xiaodi/cocogum_refactor/csn_mini_data/java_files/454451.java0")

code_wc  = pickle.load(open('/mnt/xiaodi/cocogum_refactor/csn_mini_data/vocab_raw/code_word_count_djl1_dfp0_dsi1_dlc0_dr1.pkl', 'rb'))
code_wc0 = pickle.load(open('/mnt/xiaodi/cocogum_refactor/csn_mini_data/vocab_raw/code_word_count_djl1_dfp0_dsi1_dlc1_dr1.pkl0', 'rb'))
print('code word count unchanged: ', code_wc == code_wc0)

sbt_wc  = pickle.load(open('/mnt/xiaodi/cocogum_refactor/csn_mini_data/vocab_raw/sbt_word_count.pkl', 'rb'))
sbt_wc0 = pickle.load(open('/mnt/xiaodi/cocogum_refactor/csn_mini_data/vocab_raw/sbt_word_count.pkl0', 'rb'))
print('sbt word count unchanged: ', sbt_wc == sbt_wc0)

summary_wc  = pickle.load(open('/mnt/xiaodi/cocogum_refactor/csn_mini_data/vocab_raw/summary_word_count_cfp1_csi1_cfd0_clc1.pkl', 'rb'))
summary_wc0 = pickle.load(open('/mnt/xiaodi/cocogum_refactor/csn_mini_data/vocab_raw/summary_word_count_cfp1_csi1_cfd0_clc1.pkl0', 'rb'))
print('summary word count unchanged: ', summary_wc == summary_wc0)

vocab = open_json('/mnt/xiaodi/cocogum_refactor/csn_mini_data/vocab_raw/csn_trainingset_djl1_dfp0_dsi1_dlc1_dr1_cfp1_csi1_cfd0_clc1.json')
vocab0 = open_json('/mnt/xiaodi/cocogum_refactor/csn_mini_data/vocab_raw/csn_trainingset_djl1_dfp0_dsi1_dlc1_dr1_cfp1_csi1_cfd0_clc1.json0')
print('vocab unchanged: ', vocab == vocab0)

data  = pickle.load(open('/mnt/xiaodi/cocogum_refactor/csn_mini_data/dlen100_clen12_slen435_dvoc-1_cvoc-1_svoc-1_dataset.pkl', 'rb'))
data0 = pickle.load(open('/mnt/xiaodi/cocogum_refactor/csn_mini_data/dlen100_clen12_slen435_dvoc-1_cvoc-1_svoc-1_dataset.pkl0', 'rb'))
print('processed data unchanged: ', data == data0)