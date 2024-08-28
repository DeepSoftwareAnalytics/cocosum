import pickle
import sys

sys.path.append("../")
from util.DataUtil import open_json

# check fids.
fids = pickle.load(open('/mnt/xiaodi/cocogum_refactor/csn_mini_data/correct_fid/cfd1_sfd1_ufd1_fid.pkl', 'rb'))
fids0 = pickle.load(open('/mnt/xiaodi/cocogum_refactor/csn_mini_data/correct_fid/cfd1_sfd1_ufd1_fid.pkl0', 'rb'))
print('fids unchanged: ', fids == fids0)

# check UML.
dot = open_json('/mnt/xiaodi/cocogum_refactor/csn_mini_data/umldata/dot_ref/test/454456_rewrite.dot')
dot0 =open_json('/mnt/xiaodi/cocogum_refactor/csn_mini_data/umldata/dot_ref/test/454456_rewrite.dot0')

big_graph_id = pickle.load(open('/mnt/xiaodi/cocogum_refactor/csn_mini_data/umldata/SBT_jsonl/test/big_graph_id.pkl', 'rb'))
big_graph_id0 = pickle.load(open('/mnt/xiaodi/cocogum_refactor/csn_mini_data/umldata/SBT_jsonl/test/java_test_0.jsonl/big_graph_id.pkl0', 'rb'))
print('big_graph_id unchanged: ', big_graph_id == big_graph_id0)

m2ui = pickle.load(open('/mnt/xiaodi/cocogum_refactor/csn_mini_data/umldata/SBT_jsonl/test/method2uml_index.pkl', 'rb'))
m2ui0 = pickle.load(open('/mnt/xiaodi/cocogum_refactor/csn_mini_data/umldata/SBT_jsonl/test/java_test_0.jsonl/method2uml_index.pkl0', 'rb'))
print('m2ui unchanged: ', m2ui == m2ui0)

methods = pickle.load(open('/mnt/xiaodi/cocogum_refactor/csn_mini_data/umldata/SBT_jsonl/test/methods.pkl', 'rb'))
methods0 = pickle.load(open('/mnt/xiaodi/cocogum_refactor/csn_mini_data/umldata/SBT_jsonl/test/java_test_0.jsonl/methods.pkl0', 'rb'))
print('methods unchagned: ', methods == methods0)

umls = pickle.load(open('/mnt/xiaodi/cocogum_refactor/csn_mini_data/umldata/SBT_jsonl/test/umls.pkl', 'rb'))
umls0 = pickle.load(open('/mnt/xiaodi/cocogum_refactor/csn_mini_data/umldata/SBT_jsonl/test/java_test_0.jsonl/umls.pkl0', 'rb'))
print('umls unchagned: ', umls == umls0)