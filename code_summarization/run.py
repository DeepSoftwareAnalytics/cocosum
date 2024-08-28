import os
import subprocess
import os.path
import sys

hostname = 'ccnt-ubuntu'

if hostname == 'ccnt-ubuntu':
    print(hostname)
    def preprocess():
        log = '/mnt/yuxuan/Code_summarization/csn_dataset/train_data/log.preprocess'
        if os.path.exists(log):
            os.system("rm -rf %s" % log)

        run = 'python preprocess_append_id.py ' \
              '-data_name github-javalang ' \
              '-train_src ./csn_dataset/original/code_tokens_javalang.pkl ' \
              '-train_tgt ./csn_dataset/original/docstring_tokens.pkl ' \
              '-train_xe_src ./csn_dataset/original/drl_csn.pkl ' \
              '-train_xe_tgt ./csn_dataset/original/docstring_tokens.pkl ' \
              '-train_pg_src ./csn_dataset/original/csn.pkl ' \
              '-train_pg_tgt ./csn_dataset/original/docstring_tokens.pkl ' \
              '-valid_src ./csn_dataset/original/code_tokens_javalang.pkl ' \
              '-valid_tgt ./csn_dataset/original/docstring_tokens.pkl ' \
              '-test_src ./csn_dataset/original/code_tokens_javalang.pkl ' \
              '-test_tgt ./csn_dataset/original/docstring_tokens.pkl ' \
              '-save_data ./csn_dataset/train_data/processed ' \
              '> ./csn_dataset/train_data/log.preprocess'
        print(run)
        a = os.system(run)
        if a == 0:
            print("finished.")
        else:
            print("failed.")
            sys.exit()

    def train_a2c(file_path, start_reinforce, end_epoch, critic_pretrain_epochs, data_type, has_attn, gpus):
        import time
        start_time = time.time()
        model_name = file_path.split('/')[-1]
        run = 'python a2c-train.py ' \
              '-data %s ' \
              '-save_dir ./csn_dataset/train_data_out/ ' \
              '-embedding_w2v ./csn_dataset/train_data ' \
              '-start_reinforce %s ' \
              '-end_epoch %s ' \
              '-critic_pretrain_epochs %s ' \
              '-data_type %s ' \
              '-has_attn %s ' \
              '-gpus %s ' \
              '> ./csn_dataset/train_data_out/log.a2c-train_%s_%s_%s_%s_%s_g%s_model.test' \
              % (file_path, start_reinforce, end_epoch, critic_pretrain_epochs, data_type, has_attn, gpus,
                 start_reinforce, end_epoch, critic_pretrain_epochs, data_type, has_attn, gpus)
        print(run)
        a = os.system(run)
        if a == 0:
            print("finished.")
            print(model_name)
            print('cost_time: ', time.time()-start_time)
        else:
            print("failed.")
            sys.exit()

    def test_a2c(data_type, has_attn, gpus):
        run = 'python a2c-train.py ' \
              '-data ./csn_dataset/train_data/processed_filter.train.pt2 ' \
              '-load_from ./csn_dataset/train_data_out/model_rf_hybrid_1_7_reinforce.pt ' \
              '-embedding_w2v ./csn_dataset/train_data/ ' \
              '-eval -save_dir . ' \
              '-data_type %s ' \
              '-has_attn %s ' \
              '-gpus %s ' \
              '> ./csn_dataset/train_data_out/log.a2c-test_%s_%s_%s' \
              % (data_type, has_attn, gpus, data_type, has_attn, gpus)
        print(run)
        a = os.system(run)
        if a == 0:
            print("finished.")
        else:
            print("failed.")
            sys.exit()

    if sys.argv[1] == 'preprocess':
        preprocess()

    if sys.argv[1] == 'train_a2c':

        file_path = './csn_dataset/train_data/processed_filter_tgt_10.train.pt'
        train_a2c(file_path, sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7])

    if sys.argv[1] == 'test_tree_depth':
        root_path = './csn_dataset/test_diff_tree_depth'
        for root, sub, filenames in os.walk(root_path):
            for file in filenames:
                file_path = os.path.join(root, file)
                train_a2c(file_path, sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7])

    if sys.argv[1] == 'test_a2c':
        test_a2c(sys.argv[2], sys.argv[3], sys.argv[4])