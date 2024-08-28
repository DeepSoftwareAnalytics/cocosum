import string


class Config(object):
    ###### Used in Model. ##########
    use_oov_sum = False  # False (which makes more sense: New calculation: use groudtruth summary, do not filter OOV words), True (Leclair approach)
    use_full_sum = True
    trim_til_EOS = True
    EXP = True  # True when tuning parameters
    DEBUG = True
    gpu_id = 1
    in_path = './data/csn/dataset/dlen100_clen12_slen435_dvoc10000_cvoc10000_svoc10000_dataset.pkl'
    out_path = "model.path"

    # beam search only for codenn and h-deepcom
    beam_search_method = "greedy"  # beam search strategy: bfs  greedy  none
    beam_width = 1  # > 0 then use beam search for codenn or hdeepcom

    # https://github.com/mcmillco/funcom/blob/41c737903a286e428493705e53c9c0e5946d2af6/train.py#L77
    batch_size = 256  # 200, 128 256
    modeltype = 'uml'  # 'ast-att-gru', 'att-gru', 'codenn', 'uml', 'h-deepcom', 'ast-att-transformer', 'ast-att-gru-codebert', 'uml-codebert', 'ast-att-code2vec', 'uml-transformer', 'uml-code2vec'
    pretrained_modeltype = ''  # roberta
    pretrained_modelpath = '/mnt/xiaodi/CodeSum/pretrained_codebert/'
    tokenizer_name = 'roberta-base'
    source_code_path = '/mnt/Data/csn/csn.pkl'
    random_seed = 113111
    root = 'data/codenet_uml/'

    # https://github.com/mcmillco/funcom/blob/41c737903a286e428493705e53c9c0e5946d2af6/models/ast_attendgru_xtra.py#L34
    code_dim = 128
    summary_dim = 128
    sbt_dim = 128

    rnn_hidden_size = 256  # 256, 128

    lr = 0.001  # default learning rate for Adam in Keras
    num_epochs = 40
    num_subprocesses = 4  # for dataloader, default 0
    eval_frequency = 1
    ########## Used in Model END ##########

    csn_ori_train_path = "/mnt/Data/csn/csn_ori_java/java/final/jsonl/train"
    csn_ori_val_path = "/mnt/Data/csn/csn_ori_java/java/final/jsonl/valid"
    csn_ori_test_path = "/mnt/Data/csn/csn_ori_java/java/final/jsonl/test"

    data_dir = "/mnt/Data/csn/data_csn_allvoc_d100_c13"  # data_len = 100, summary_len = 13

    PAD_token_id = 0  # <NULL>
    SOS_token_id = 1  # <s>
    EOS_token_id = 2  # </s>
    UNK_token_id = None  # <UNK> oov_index is comvocabsize/datvocabsize/smlvocabsize - 1

    PAD_token = "<NULL>"
    SOS_token = "<s>"
    EOS_token = "</s>"
    UNK_token = "<UNK>"

    # seq len
    code_len = 100  # 256 220 150 100 80 50
    summary_len = 12  # 64 50 35 22 16 10
    sbt_len = 435

    # voc size
    # use -1 if you want to include all the tokens
    code_vocab_size = 10000  # 1000, 2000, 3000,
    summary_vocab_size = 10000
    sbt_vocab_size = 10000

    code_tokens_using_javalang_results = True
    code_filter_punctuation = False  # True False
    code_split_identifier = True
    code_lower_case = True  # True False
    code_replace_string_num = True

    summary_filter_punctuation = True  #
    summary_split_identifier = True  #
    summary_filter_bad_cases = True

    uml_filter_data_bad_cases = True

    # Used in preprocess. Not in the model. ###########
    # STOP_WORDS = set([i for i in string.punctuation] + [str(i) for i in range(10)] + ['<p'])
    STOP_WORDS = set([i for i in string.punctuation] + ['<p'])

    csn_ori_java_path = "/mnt/Data/csn/csn_ori_java/java/final/jsonl/"
    processed_data_path = "../../data/csn/"

    # processed_data_path = '/mnt/xiaodi/cocogum_refactor/csn_mini_data/'
    # vocab_path = os.path.join('../../../Data/csn/vocab_raw', voc_info_file_name)
    dataset_partitions = ['train', 'val', 'test']
    # dataset_path = "/mnt/Data/csn/dataset"
    # dataset_path = '/mnt/xiaodi/cocogum_refactor/csn_mini_data'
    dataset_path = "../../data/csn/dataset"
    # codenn
    codenn_hidden_size = 256

    ##### uml ###############
    docstring_tokens_split_path = "./data/csn/summary/cfp1_csi1_cfd0_clc1.pkl"
    m2u_m2c_path = "../../data/csn/uml/m2u_m2c.pkl"
    uml_path = "./data/csn/uml/uml_dataset.pt"
    gnn_num_feature = 513
    gnn_num_hidden = [128, 256]
    gnn_dropout = [0.6, 0.6, 0.6]
    # gnn_num_hidden = [8]
    # gnn_dropout = [0.0, 0.0]
    decoder_type = "ASTAttGRU_AttTwoChannelTrans"  # ASTAttGRU_BertAtt, ASTAttGRU_AttTwoChannel, ASTAttGRU_AttTwoChannelTrans, ASTAttGRU_UML, ASTAttGRU_Gate. ASTAttGRU_BertAstAtt
    gnn_type = "HGCN"  # "GAT"  "HGAT" "GCN" "HGCN"
    gnn_concat_uml = False
    gnn_concat = True
    # aggregation in ["Mean", "Concat"]
    aggregation = "Mean"
    enable_sbt = True
    enable_uml = True
    # class_only = True will disable gnn_type, gnn_concat_uml and gnn_concat
    class_only = False
    # enable_func please set False (haven't implement)
    enable_func = False
    num_func_feature = 512
    weight_decay = 0.3

    rnn_dropout = 0
    rnn_num_layers = 1
    rnn_bidirectional = False
    schedual_sampling_prob = 0
    ###### uml end ##############

    package_wise = False
    method_wise = False

    summary_tokens_file_name = "cfp" + str(summary_filter_punctuation + 0) + \
                               "_csi" + str(summary_split_identifier + 0) + \
                               "_cfd0_clc1.pkl"
    code_tokens_file_name = "djl" + str(code_tokens_using_javalang_results + 0) + \
                            "_dfp" + str(code_filter_punctuation + 0) + \
                            "_dsi" + str(code_split_identifier + 0) + \
                            "_dlc" + str(code_lower_case + 0) + \
                            "_dr" + str(code_replace_string_num + 0) + ".pkl"
    sbt_tokens_file_name = 'sbts_tokens.pkl'
    voc_info_file_name = 'csn_trainingset_' + \
                         code_tokens_file_name.split(".")[0] + "_" + \
                         summary_tokens_file_name.split(".")[0] + '.json'
    multi_gpu = False
