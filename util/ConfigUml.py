
class Config(object):
    ########## Used in Model. ##########
    trimTilEOS = False
    EXP = False   # True when tuning parameters
    DEBUG = False
    gpu_id = 2
    # in_path = "./data/my_sbt_small_voc1000/dataset.pkl"
    # in_path = "./data/my_sbt_small/dataset.pkl"
    # in_path = "./data/my_sbt_all_voc1000/dataset.pkl"
    # in_path = "./data/my_sbt_all/dataset.pkl"
    # in_path = "./data/small_funcom_with_csn_size/dataset.pkl"
    # in_path = "./data_leclair/standard_data.pkl"
    #in_path = "./data/1k/dataset_uml.pkl"
    in_path = "./data/csn/dataset_uml.pkl"
    docstring_tokens_split_path = "data/csn/docstring_tokens_split.pkl"
    
    random_seed = 113111
    uml_path = "data/csn/uml_dataset.pt"
    gnn_num_feature = 513
    gnn_num_hidden = [256, 256]
    gnn_dropout = [0.6, 0.6, 0.6]
    gnn_type = "HGAT"
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
    
    # out_path = "model.pth"  # If not empty, then output model
    out_path = "model_sbt_uml.pt"

    # https://github.com/mcmillco/funcom/blob/41c737903a286e428493705e53c9c0e5946d2af6/train.py#L77
    # batch_size = 200
    batch_size = 256
    # batch_size = 16
    

    # https://github.com/mcmillco/funcom/blob/41c737903a286e428493705e53c9c0e5946d2af6/models/ast_attendgru_xtra.py#L34
    # code_dim = 100
    # summary_dim = 100
    # sbt_dim = 10
    code_dim = 128
    summary_dim = 128
    sbt_dim = 128

    rnn_hidden_size = 256
    # rnn_hidden_size = 128

    lr = 0.001     # default learning rate for Adam in Keras
    num_epochs = 70
    num_subprocesses = 4  # for dataloader, default 0
    eval_frequency = 1
    ########## Used in Model. ##########

    ########## Used in preprocess. Not in the model. ###########
    train_data_path = "/datadrive/CodeSummary_Data/raw_data/CodeSearchNet/java/final/jsonl/train"
    val_data_path = "/datadrive/CodeSummary_Data/raw_data/CodeSearchNet/java/final/jsonl/valid"
    test_data_path = "/datadrive/CodeSummary_Data/raw_data/CodeSearchNet/java/final/jsonl/test"

    # data_dir = "./data/code_search_net_all_voc"
    # data_dir = "./data/code_search_net_voc1000"
    data_dir = "./data/data_csn_allvoc_d100_c13"  # data_len = 100, summary_len = 13

    PAD_token_id = 0     # <NULL>
    SOS_token_id = 1     # <s>
    EOS_token_id = 2     # </s>
    UNK_token_id = None  # <UNK> oov_index is comvocabsize/datvocabsize/smlvocabsize - 1

    PAD_token = "<NULL>"
    SOS_token = "<s>"
    EOS_token = "</s>"
    UNK_token = "<UNK>"


    ## For CodeSearchNet ##
    code_len = 100
    summary_len = 22
    sbt_len = 100

    # use -1 if you want to include all the tokens
    code_vocab_size = -1
    summary_vocab_size = -1
    ## For CodeSearchNet ##

    ########## Used in preprocess. Not in the model. ###########
