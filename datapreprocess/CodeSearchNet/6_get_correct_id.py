import sys
import os
import time
sys.path.append('../../')
from util.Config import Config as config
from util.DataUtil import read_pickle_data, save_pickle_data, time_format


def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except (UnicodeDecodeError, UnicodeEncodeError):
        return False
    else:
        return True


class Config:
    dataset_path = '../../data/csn/'


def correct_summary_fids(data):
    gt3_fids = [fid for fid, item in data.items() if len(item) > 3]
    english_fids = [fid for fid, item in data.items() if isEnglish(" ".join(item))]
    return list(set(gt3_fids).intersection(set(english_fids)))


def correct_code_fids(data):
    return [fid for fid, item in data.items() if isEnglish(" ".join(item))]


if __name__ == '__main__':

    # summary = read_pickle_data('../../csn_mini_data/summary/cfp1_csi1_cfd0_clc1.pkl')
    # code = read_pickle_data('../../csn_mini_data/code/djl1_dfp0_dsi1_dlc1_dr1.pkl')
    # sbt = read_pickle_data('../../csn_mini_data/sbt/sbts.pkl')
    # uml = read_pickle_data('../../UmlEmbeddingTools/data/csn_mini/' + part + '/methods.pkl')
    start_time = time.perf_counter()
    summary = read_pickle_data(os.path.join(Config.dataset_path, 'summary/cfp1_csi1_cfd0_clc1.pkl'))
    code = read_pickle_data(os.path.join(Config.dataset_path, 'code/djl1_dfp0_dsi1_dlc1_dr1.pkl'))
    sbt = read_pickle_data(os.path.join(Config.dataset_path, 'sbt/sbts.pkl'))
    correct_uml_fid = read_pickle_data(os.path.join(Config.dataset_path, 'uml/correct_uml_fid.pkl'))

    final_correct_id = {}
    for part in summary:
        summary_id = correct_summary_fids(summary[part])
        code_id = correct_code_fids(code[part])
        sbt_id = list(sbt[part].keys())
        uml_id = correct_uml_fid[part]
        final_correct_id[part] = list(set(summary_id).intersection(set(code_id)).
                                      intersection(set(sbt_id)).intersection(set(uml_id)))
        print(part + "len final_correct_id ", len( final_correct_id[part] ))
        # tmp1 = list(set(summary_id).intersection(set(code_id)).intersection(set(sbt_id)).intersection(set(uml_id)))
        # tmp2 = list(set(tmp1).intersection(set(sbt_id)))
        # correct_id = list(set(tmp2).intersection(set(uml_id)))
        # final_correct_id[part] = correct_id
    #
    # correct_id_filename = "cfd" + str(config.summary_filter_bad_cases + 0) + \
    #                       "_sfd1" + "_ufd" + str(config.uml_filter_data_bad_cases + 0) + "_fid.pkl"
    correct_id_filename = "cfd1_sfd1_ufd1_fid.pkl"
    save_pickle_data(os.path.join(Config.dataset_path, "correct_fid"), correct_id_filename, final_correct_id)
    print("time cost %s" % time_format(time.perf_counter() - start_time))
