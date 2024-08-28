import sys
import os
sys.path.append('../../')
from util.Config import Config as config
from util.DataUtil import read_pickle_data, save_pickle_data

summary = read_pickle_data('../../csn_mini_data/summary/cfp1_csi1_cfd0_clc1.pkl')
code = read_pickle_data('../../csn_mini_data/code/djl1_dfp0_dsi1_dlc0_dr1.pkl')
sbt = read_pickle_data('../../csn_mini_data/sbt/sbts.pkl')
final_id = {}
for part in summary:
    summary_id = list(summary[part].keys())
    code_id = list(code[part].keys())
    sbt_id = list(sbt[part].keys())
    uml = read_pickle_data('../../UmlEmbeddingTools/data/csn_mini/' + part + '/methods.pkl')
    uml_id = list(uml.keys())

    tmp1 = list(set(summary_id).intersection(set(code_id)))
    tmp2 = list(set(tmp1).intersection(set(sbt_id)))
    correct_id = list(set(tmp2).intersection(set(uml_id)))
    final_id[part] = correct_id

correct_id_filename = "cfd" + str(config.summary_filter_bad_cases + 0) +\
                          "_sfd1" + "_ufd" + str(config.uml_filter_data_bad_cases + 0) + "_fid.pkl"

save_pickle_data(os.path.join(config.processed_data_path, "correct_fid"), correct_id_filename, final_id)