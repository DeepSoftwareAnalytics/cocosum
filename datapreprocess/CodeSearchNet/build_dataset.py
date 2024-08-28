# refer to 13.76.197.150/notebooks/mnt/Data/csn/process_multivoc_datasets.ipynb

import sys
sys.path.append("../../")
import os
from util.DataUtil import save_pickle_data, padding, read_pickle_data, load_vocab,\
    set_config_data_processing, basic_info_logger_data_processing, get_file_name
from util.Config import Config as config
import time
from util.LoggerUtil import set_logger


def code2ids(tokens, w2i, seq_len):
    unk_id = w2i[config.UNK_token]
    ids = [w2i.get(token, unk_id) for token in tokens[:seq_len]]
    ids = padding(ids, seq_len, config.PAD_token_id)
    return ids


def summary2ids(summary_tokens, summary_w2i):
    summary_unk_id = summary_w2i[config.UNK_token]
    summary_ids = [summary_w2i.get(token, summary_unk_id) for token in summary_tokens[:config.summary_len - 1]]
    summary_ids.insert(0, config.SOS_token_id)
    if len(summary_ids) < config.summary_len:
        summary_ids.append(config.EOS_token_id)
    summary_ids = padding(summary_ids, config.summary_len, config.PAD_token_id)
    return summary_ids


def process_data(summary, code, sbt, code_w2i, summary_w2i, sbt_w2i, fids):
    code_ids = {}
    summary_ids = {}
    sbt_ids = {}

    for fid in fids:
        code_ids[fid] = code2ids(code[fid], code_w2i, config.code_len)
        summary_ids[fid] = summary2ids(summary[fid], summary_w2i)
        sbt_ids[fid] = code2ids(sbt[fid], sbt_w2i, config.sbt_len)
    return code_ids, summary_ids, sbt_ids


def get_dataset(vocab_size, enable_uml=True):
    code_word_count, summary_word_count, sbt_word_count, \
        summary_w2i, summary_i2w, code_i2w, code_w2i, \
        sbt_i2w, sbt_w2i = load_vocab(os.path.join(config.processed_data_path, "vocab_raw", config.voc_info_file_name), vocab_size)

    summary_tokens = read_pickle_data(os.path.join(config.processed_data_path, "summary", config.summary_tokens_file_name))
    code_tokens = read_pickle_data(os.path.join(config.processed_data_path, "code", config.code_tokens_file_name))
    sbt_tokens = read_pickle_data(os.path.join(config.processed_data_path, "sbt", config.sbt_tokens_file_name))

    correct_id_filename = "cfd{}_sfd1_ufd{}_fid.pkl".format(str(config.summary_filter_bad_cases + 0), str(config.uml_filter_data_bad_cases + 0))
    correct_fid_path = os.path.join(config.processed_data_path, "correct_fid", correct_id_filename)
    correct_fid = read_pickle_data(correct_fid_path)

    dtrain, ctrain, strain = process_data(summary_tokens["train"],  code_tokens["train"],
                                   sbt_tokens["train"], code_w2i, summary_w2i, sbt_w2i,  correct_fid["train"])
    dval, cval, sval = process_data(summary_tokens["val"],  code_tokens["val"],
                                  sbt_tokens["val"], code_w2i, summary_w2i, sbt_w2i, correct_fid["val"])
    dtest, ctest, stest = process_data(summary_tokens["test"],  code_tokens["test"],
                                sbt_tokens["test"], code_w2i, summary_w2i, sbt_w2i, correct_fid["test"])

    dataset = {"ctrain": ctrain, "cval": cval, "ctest": ctest,
               "dtrain": dtrain, "dval": dval, "dtest": dtest,
               "strain": strain, "sval": sval, "stest": stest,
               "comstok": {"i2w": summary_i2w, "w2i": summary_w2i, "word_count": summary_word_count},
               "datstok": {"i2w": code_i2w, "w2i": code_w2i, "word_count": code_word_count},
               "smlstok": {"i2w": sbt_i2w, "w2i": sbt_w2i, "word_count": sbt_word_count},
               "config": {"datvocabsize": len(code_i2w), "comvocabsize": len(summary_i2w),
                          "smlvocabsize": len(sbt_i2w), "datlen": config.code_len,
                          "comlen": config.summary_len, "smllen": config.sbt_len}}

    if enable_uml:
        m2u_m2c = read_pickle_data(config.m2u_m2c_path)
        dataset.update(m2u_m2c)

    return dataset


if __name__ == "__main__":

    set_config_data_processing()
    set_logger(config.DEBUG)
    basic_info_logger_data_processing()
    start = time.perf_counter()
    vocab_size = {"code": config.code_vocab_size, "summary":
        config.summary_vocab_size, "sbt": config.sbt_vocab_size}
    dataset = get_dataset(vocab_size, config.enable_uml)
    filename = get_file_name()
    # config.dataset_path = os.path.join(config.processed_data_path, "dataset_uml")
    save_pickle_data(config.dataset_path, filename, dataset)
    # print(filename)
    print('time: ', time.perf_counter() - start)



