import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from loadModel import load_summary
from models.ASTAttGRU import AstAttGRUModel
from models.AttGRU import AttGRUModel
from models.HDeepCom import HDeepComModel
from models.codenn import CodeNNModel
from models.ASTAttTransformer import AstAttTransformerModel
from models.ASTAttTransformer_AttTwoChannelTrans import AstAttTransformerModelUML
from models.CodeNN_AttTwoChannelTrans import CodeNNModelUML
from models.HDeepCom_AttTwoChannelTrans import HDeepComModelUML
from util.Config import Config as cf
from util.DataUtil import read_pickle_data, set_seed, make_directory, str_to_bool, get_file_name, \
    read_funcom_format_data
from util.Dataset import CodeSummaryUmlDataset
from util.EvaluateUtil import calculate_bleu_uml, calculate_bleu, metetor_rouge_cider
from util.GPUUtil import move_model_to_device, move_to_device, move_pyg_to_device
from util.GPUUtil import set_device
from util.LoggerUtil import info_logger, set_logger, debug_logger, torch_summarize, count_parameters
from loadModel import load_model
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


def train_seq(model, train_data_loader, optimizer, loss_fn):
    model.train()
    cumulative_loss = 0
    step_size = 0
    for train_batch_data in train_data_loader:
        for i in range(len(train_batch_data)):
            train_batch_data[i] = move_to_device(train_batch_data[i])
        # zero all of the gradients
        optimizer.zero_grad()
        if cf.modeltype == 'ast-att-gru':
            output = model(method_code=train_batch_data[0], method_sbt=train_batch_data[1],
                           method_summary=train_batch_data[2], use_teacher=True)[-1]
        elif cf.modeltype == "ast-att-transformer":
            output = model(method_code=train_batch_data[0], method_sbt=train_batch_data[1],
                           method_summary=train_batch_data[2], use_teacher=True)[-1]
        elif cf.modeltype == 'att-gru':
            output = model(method_code=train_batch_data[0], method_summary=train_batch_data[2], use_teacher=True)[-1]
        elif cf.modeltype == 'codenn':
            output = model(method_code=train_batch_data[0], beam_width=cf.beam_width, is_test=False)[-1]
        elif cf.modeltype == 'h-deepcom':
            output = model(method_code=train_batch_data[0], method_sbt=train_batch_data[1],
                           beam_width=cf.beam_width, is_test=False)[-1]
        else:
            raise Exception("Unrecognized Model: ", str(cf.modeltype))

        sum_vocab_size = output.shape[-1]

        # output: batch_size, summary_length - 1, sum_vocab_size
        output = output.view(-1, sum_vocab_size)
        # output: batch_size * (summary_length - 1), sum_vocab_size
        # exclude <s>
        trg = train_batch_data[-1][:, 1:].reshape(-1)
        # trg = [batch size * (summary_length - 1)]
        loss = loss_fn(output, trg)
        # Backward pass
        loss.backward()

        if cf.modeltype == 'codenn':
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        # Calling the step function on an Optimizer makes an update to its
        # parameters
        optimizer.step()

        # calculate loss
        cumulative_loss += loss.item()
        step_size += 1

    return cumulative_loss / step_size


def train_uml(model, train_data_loader, optimizer, loss_fn, uml_data):
    model.train()

    cumulative_loss = 0
    step_size = 0
    for idx, train_batch_data in enumerate(train_data_loader):
        for i in range(len(train_batch_data) - 2):
            train_batch_data[i] = move_to_device(train_batch_data[i])
        
        # Construct uml batch
        m2u = train_batch_data[-2]
        m2c = train_batch_data[-1]
        
        uml_dict = {}
        uml_list = []
        class_st_idx = [0]
        # uml_index storage the mapping between the method and the corresponding uml (class) idx
        uml_index = []
        cnt = 0

        for it in m2u:
            i = int(it)
            if i not in uml_dict:
                uml_dict[i] = cnt
                cnt += 1
                uml_list.append(uml_data[i])
                class_st_idx.append(class_st_idx[-1] + uml_data[i].num_nodes)
            uml_index.append(uml_dict[i])

        uml_index = torch.LongTensor(uml_index)
                                     
        uml_batch = Batch.from_data_list(uml_list)
        assert uml_batch.num_nodes == class_st_idx[-1]
        
        # class_index storage the mapping between the method and the corresponding node (class) idx  
        class_index = [class_st_idx[uml_dict[int(m2u[idx])]] + it for idx, it in enumerate(m2c)]
        class_index = torch.LongTensor(class_index)

        uml_batch = move_pyg_to_device(uml_batch)
        uml_index = move_to_device(uml_index)
        class_index = move_to_device(class_index)
        
        # zero all of the gradients
        optimizer.zero_grad()
        if cf.modeltype == 'uml-transformer':
            output = model(method_code=train_batch_data[0], method_sbt=train_batch_data[1],
                           method_summary=train_batch_data[2], use_teacher=True,
                           uml_data=uml_batch, class_index=class_index, uml_index=uml_index)[-1]
        elif cf.modeltype == 'uml-code-nn':
            output = model(method_code=train_batch_data[0], beam_width=cf.beam_width, is_test=False,
                           uml_data=uml_batch, class_index=class_index, uml_index=uml_index)[-1]
        elif cf.modeltype == 'uml-h-deepcom':
            output = model(method_code=train_batch_data[0], method_sbt=train_batch_data[1],
                           beam_width=cf.beam_width, is_test=False, uml_data=uml_batch,
                           class_index=class_index, uml_index=uml_index)[-1]
        elif cf.modeltype == 'uml':
            output = model(method_code=train_batch_data[0], method_sbt=train_batch_data[1],
                           method_summary=train_batch_data[2], use_teacher=True,
                           uml_data=uml_batch, class_index=class_index, uml_index=uml_index)[-1]
        # if cf.enable_sbt:
        #     output = model(method_code=train_batch_data[0], method_sbt=train_batch_data[1],
        #                    method_summary=train_batch_data[2], use_teacher=True,
        #                    uml_data=uml_batch, class_index=class_index, uml_index=uml_index)[-1]
        # else:
        #     output = model(method_code=train_batch_data[0], method_summary=train_batch_data[1],
        #                    use_teacher=True, uml_data=uml_batch, class_index=class_index, uml_index = uml_index)[-1]
            
        sum_vocab_size = output.shape[-1]

        # output: batch_size, summary_length - 1, sum_vocab_size
        output = output.view(-1, sum_vocab_size)
        # output: batch_size * (summary_length - 1), sum_vocab_size
        # exclude <s>
        if cf.enable_sbt:
            trg = train_batch_data[2]
        else:
            trg = train_batch_data[1]
        trg = trg[:, 1:].reshape(-1)
        # trg = [batch size * (summary_length - 1)]

        loss = loss_fn(output, trg)
        # Backward pass
        loss.backward()
        if cf.modeltype == 'uml-code-nn':
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        # Calling the step function on an Optimizer makes an update to its
        # parameters
        optimizer.step()

        # calculate loss
        cumulative_loss += loss.item()
        step_size += 1

    return cumulative_loss / step_size


def create_model(model_type, token_vocab_size, sbt_vocab_size, summary_vocab_size, summary_len, enable_sbt, enable_uml):
    if model_type == 'att-gru':
        # basic attention GRU model based on Nematus architecture
        return AttGRUModel(token_vocab_size, summary_vocab_size)
    elif model_type == 'ast-att-gru':
        # attention GRU model with added AST information from srcml.
        return AstAttGRUModel(token_vocab_size, sbt_vocab_size, summary_vocab_size)
    elif model_type == 'ast-att-transformer':
        return AstAttTransformerModel(token_vocab_size, sbt_vocab_size, summary_vocab_size)
    elif model_type == 'codenn':
        # attention LSTM model, refer to ACL 2016 paper.
        return CodeNNModel(token_vocab_size, summary_vocab_size, summary_len)
    elif model_type == 'h-deepcom':
        return HDeepComModel(token_vocab_size, sbt_vocab_size, summary_vocab_size, summary_len)
    elif model_type == 'uml':
        module = __import__("models." + cf.decoder_type, fromlist=["models"])
        return module.AstAttGRUModelUML(token_vocab_size, sbt_vocab_size, summary_vocab_size, enable_sbt, enable_uml)
    elif model_type == 'uml-transformer':
        return AstAttTransformerModelUML(token_vocab_size, sbt_vocab_size, summary_vocab_size)
    elif model_type == 'uml-code-nn':
        return CodeNNModelUML(token_vocab_size, summary_vocab_size, summary_len)
    elif model_type == 'uml-h-deepcom':
        return HDeepComModelUML(token_vocab_size, sbt_vocab_size, summary_vocab_size, summary_len)
    else:
        raise Exception("Unrecognized Model: ", str(model_type))


def set_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('-root', type=str, )
    parser.add_argument('-data', type=str, )
    parser.add_argument('-source_code_path', type=str, default='csn.pkl')
    parser.add_argument('-docstring_tokens_split_path', default="summary/cfp1_csi1_cfd0_clc1.pkl")
    parser.add_argument('-pretrained_modeltype', required=False)
    parser.add_argument('-pretrained_modelpath', default=" pretrained_codebert/")
    parser.add_argument('-m2u_m2c_path', default="uml/m2u_m2c.pkl")
    parser.add_argument('-uml_path', default="uml/uml_dataset.pt")

    parser.add_argument('-gpu_id', required=False)

    parser.add_argument('-batch_size', required=False)
    parser.add_argument('-modeltype', required=False)
    parser.add_argument('-code_dim', required=False)
    parser.add_argument('-summary_dim', required=False)
    parser.add_argument('-sbt_dim', required=False)
    parser.add_argument('-rnn_hidden_size', required=False)
    parser.add_argument('-lr', required=False)
    parser.add_argument('-num_epochs', required=False)
    parser.add_argument('-out_path', required=False)

    # code processing  / 5
    parser.add_argument('-djl', "--code_tokens_javalang_results", type=str, choices=["True", "False"], required=False)
    parser.add_argument('-dfp', "--code_filter_punctuation", type=str, choices=["True", "False"], required=False)
    parser.add_argument('-dsi', "--code_split_identifier", type=str, choices=["True", "False"], required=False)
    parser.add_argument('-dlc', "--code_lower_case", type=str, choices=["True", "False"], required=False)
    parser.add_argument('-dr', "--code_replace_string_num", type=str, choices=["True", "False"], required=False)

    # summary processing/3
    parser.add_argument('-cfp', "--summary_filter_punctuation", type=str, choices=["True", "False"], required=False)
    parser.add_argument('-csi', "--summary_split_identifier", type=str, choices=["True", "False"], required=False)
    parser.add_argument('-cfd', "--summary_filter_bad_cases", type=str, choices=["True", "False"], required=False)

    # seq len /3
    parser.add_argument('-dlen', "--code_len",  required=False)
    parser.add_argument('-clen', "--summary_len", required=False)
    parser.add_argument('-slen', "--sbt_len", required=False)

    # voc size /3
    parser.add_argument('-dvoc', "--code_vocab_size", required=False)
    parser.add_argument('-cvoc', "--summary_vocab_size", required=False)
    parser.add_argument('-svoc', "--sbt_vocab_size", required=False)

    # dataset
    parser.add_argument('-dataset_path', type=str, required=False)

    # gnn
    parser.add_argument('-gty', "--gnn_type", required=False)
    parser.add_argument('-gdp', "--gnn_dropout", required=False)
    parser.add_argument('-gfn', "--gnn_function_name", required=False)
    parser.add_argument('-gpp', "--gnn_project_pooling", required=False)
    parser.add_argument('-gfc', "--gnn_feature_concat", required=False)
    parser.add_argument('-wd', "--weight_decay", required=False)
    parser.add_argument('-gss', "--gnn_schedule_sampling", required=False)
    parser.add_argument('-gnf', "--gnn_num_features", required=False)
    parser.add_argument('-gnh', "--gnn_num_hidden", required=False)
    parser.add_argument('-rnl', "--rnn_num_layers", required=False)

    # beam search
    parser.add_argument('-beam_search_method', required=False)
    parser.add_argument('-beam_width', required=False)

    #  is package wise
    parser.add_argument('-pkg', "--package_wise", type=str, choices=["True", "False"], required=False)

    #  is method wise
    parser.add_argument('-mtd', "--method_wise", type=str, choices=["True", "False"], required=False)

    # aggregation in ["Mean", "Concat"]
    parser.add_argument('-agg', "--aggregation", type=str,  required=False)

    # class_only = True will disable gnn_type, gnn_concat_uml and gnn_concat
    parser.add_argument('-clo', "--class_only", type=str, choices=["True", "False"],  required=False)

    parser.add_argument('-dty', "--decoder_type", type=str, default="ASTAttGRU_AttTwoChannelTrans")

    global args
    args = parser.parse_args()
    if args.gpu_id:
        cf.gpu_id = int(args.gpu_id)
    # if args.root:
    #     cf.root = args.root
    #     cf.in_path = os.path.join(args.root, args.data)
    #     cf.docstring_tokens_split_path = args.root+args.docstring_tokens_split_path
    #     cf.m2u_m2c_path = args.root+args.m2u_m2c_path
    #     cf.uml_path = os.path.join(args.root, args.uml_path)
    #     cf.source_code_path = args.root+args.source_code_path

    if args.batch_size:
        cf.batch_size = int(args.batch_size)
    if args.modeltype:
        cf.modeltype = args.modeltype
    if args.code_dim:
        cf.code_dim = int(args.code_dim)
    if args.sbt_dim:
        cf.sbt_dim = int(args.sbt_dim)
    if args.summary_dim:
        cf.summary_dim = int(args.summary_dim)
    if args.rnn_hidden_size:
        cf.rnn_hidden_size = int(args.rnn_hidden_size)
    if args.lr:
        cf.lr = float(args.lr)
    if args.out_path:
        cf.out_path = args.out_path
    if args.num_epochs:
        cf.num_epochs = int(args.num_epochs)
    if args.pretrained_modeltype:
        cf.pretrained_modeltype = args.pretrained_modeltype


    # code processing
    if args.code_tokens_javalang_results:
        cf.code_tokens_using_javalang_results = str_to_bool(args.code_tokens_javalang_results)
    if args.code_filter_punctuation:
        cf.code_filter_punctuation = str_to_bool(args.code_filter_punctuation)
    if args.code_split_identifier:
        cf.code_split_identifier = str_to_bool(args.code_split_identifier)
    if args.code_lower_case:
        cf.code_lower_case = str_to_bool(args.code_lower_case)
    if args.code_replace_string_num:
        cf.code_replace_string_num = str_to_bool(args.code_replace_string_num)

    # summary processing
    if args.summary_filter_punctuation:
        cf.summary_filter_punctuation = str_to_bool(args.summary_filter_punctuation)
    if args.summary_split_identifier:
        cf.summary_split_identifier = str_to_bool(args.summary_split_identifier)
    if args.summary_filter_bad_cases:
        cf.summary_filter_bad_cases = str_to_bool(args.summary_filter_bad_cases)

    # seq len
    if args.code_len:
        cf.code_len = int(args.code_len)
    if args.summary_len:
        cf.summary_len = int(args.summary_len)
    if args.sbt_len:
        cf.sbt_len = int(args.sbt_len)

    # voc size
    if args.code_vocab_size:
        cf.code_vocab_size = int(args.code_vocab_size)
    if args.summary_vocab_size:
        cf.summary_vocab_size = int(args.summary_vocab_size)
    if args.sbt_vocab_size:
        cf.sbt_vocab_size = int(args.sbt_vocab_size)

    # dataset
    # if args.dataset_path:
    #     cf.dataset_path = args.dataset_path

    # gnn
    if args.gnn_type:
        cf.gnn_type = args.gnn_type
    if args.gnn_dropout:
        gdp = float(args.gnn_dropout)
        cf.gnn_dropout = [gdp, gdp, gdp]
    if args.gnn_function_name:
        cf.enable_func = str_to_bool(args.gnn_function_name)
    if args.gnn_project_pooling:
        cf.gnn_concat_uml = str_to_bool(args.gnn_project_pooling)
    if args.gnn_feature_concat:
        cf.gnn_concat = str_to_bool(args.gnn_feature_concat)
    if args.weight_decay:
        cf.weight_decay = float(args.weight_decay)
    if args.gnn_schedule_sampling:
        cf.schedual_sampling_prob = float(args.gnn_schedule_sampling)
    if args.gnn_num_features:
        cf.gnn_num_feature = int(args.gnn_num_features)
    if args.gnn_num_hidden:
        gnh = int(args. gnn_num_hidden)
        cf.gnn_num_hidden = [gnh, gnh]
    if args.rnn_num_layers:
        cf.rnn_num_layers = args.rnn_num_layers

    filename = get_file_name()
    # cf.in_path = os.path.join(cf.dataset_path, filename)
    if args.dataset_path:
        cf.dataset_path = args.dataset_path

        cf.in_path = os.path.join(cf.dataset_path, filename)
    if args.package_wise:
        cf. package_wise = str_to_bool(args.package_wise)

    if args.method_wise:
        cf.method_wise = str_to_bool(args.method_wise)

    # # aggregation in ["Mean", "Concat"]
    if args.aggregation:
        cf.aggregation = args.aggregation

    # class_only = True will disable gnn_type, gnn_concat_uml and gnn_concat
    if args.class_only:
        cf.class_only = str_to_bool(args.class_only)
    if args.decoder_type:
        cf.decoder_type = str(args.decoder_type)

    if cf. package_wise:
        cf.docstring_tokens_split_path = "/mnt/Data/csn/package_wise/summary/cfp1_csi1_cfd0_clc1.pkl"
        cf.m2u_m2c_path = "../../../Data/csn/package_wise/uml/m2u_m2c.pkl"
        cf.uml_path = "/mnt/Data/csn/package_wise/uml/uml_dataset.pt"

    if cf. method_wise:
        cf.docstring_tokens_split_path = "/mnt/Data/csn/method_wise/summary/cfp1_csi1_cfd0_clc1.pkl"
        cf.m2u_m2c_path = "../../../Data/csn/method_wise/uml/m2u_m2c.pkl"
        cf.uml_path = "/mnt/Data/csn/method_wise/uml/uml_dataset.pt"

    cf.out_path = "mt" + str(cf.modeltype) + "_bs" + str(cf.batch_size) + \
                  "_ddim" + str(cf.code_dim) + "_cdim" + str(cf.summary_dim) + "_sdim" + str(cf.sbt_dim) + \
                  "_hdim" + str(cf.rnn_hidden_size) + "_lr" + str(cf.lr) + "_gty" + str(cf.gnn_type) + "_gdp" + \
                  "_".join('%s' % id for id in cf.gnn_dropout) + '-gfn' + str(cf.enable_func + 0) + '-gpp' + str(cf.gnn_concat_uml + 0) + \
                  '_gfc' + str(cf.gnn_concat + 0) + '_wd' + str(cf.weight_decay) + '_gss' + \
                   str(cf.schedual_sampling_prob) + '_gnf' + str(cf.gnn_num_feature) + \
                  "_gnh" + "_".join('%s' % id for id in cf.gnn_num_hidden) + "-agg" + str(cf.aggregation) + "-clo" + str(cf.class_only) + \
                  "_dty" + str(cf.decoder_type) + time.strftime("%Y%m%d%H%M%S")



    # writer = SummaryWriter(os.path.join(cf.out_path, 'log'))
    # global writer
    cf.out_path = cf.out_path + ".pt"
    

def basic_info_logger():
    info_logger("[Setting] EXP: %s" % (str(cf.EXP)))
    info_logger("[Setting] DEBUG: %s" % (str(cf.DEBUG)))
    info_logger("[Setting] trim_til_EOS: %s" % (str(cf.trim_til_EOS)))
    info_logger("[Setting] use_full_sum: %s" % (str(cf.use_full_sum)))
    info_logger("[Setting] use_oov_sum: %s" % (str(cf.use_oov_sum)))
    info_logger("[Setting] Method: %s" % cf.modeltype)
    info_logger("[Setting] in_path: %s" % cf.in_path)
    info_logger("[Setting] GPU id: %d" % cf.gpu_id)
    info_logger("[Setting] num_epochs: %d" % cf.num_epochs)
    info_logger("[Setting] batch_size: %d" % cf.batch_size)
    info_logger("[Setting] code_dim: %d" % cf.code_dim)
    info_logger("[Setting] sbt_dim: %d" % cf.sbt_dim)
    info_logger("[Setting] summary_dim: %d" % cf.summary_dim)
    info_logger("[Setting] rnn_hidden_size: %d" % cf.rnn_hidden_size)
    info_logger("[Setting] lr: %f" % cf.lr)
    info_logger("[Setting] num_subprocesses: %d" % cf.num_subprocesses)
    info_logger("[Setting] eval_frequency: %d" % cf.eval_frequency)
    info_logger("[Setting] out_name: %s" % cf.out_path)
    info_logger("[Setting] decoder_type: %s" % cf.decoder_type)

    if cf.modeltype == 'uml':
        info_logger("[Setting] gnn_num_features: %d" % cf.gnn_num_feature)
        info_logger("[Setting] gnn_num_hidden: " + str(cf.gnn_num_hidden))
        info_logger("[Setting] gnn_dropout: " + str(cf.gnn_dropout))
        info_logger("[Setting] enable_sbt: " + str(cf.enable_sbt))
        info_logger("[Setting] gnn_type:" + cf.gnn_type)
        info_logger("[Setting] weight_decay:" + str(cf.weight_decay))
        info_logger("[Setting] rnn_dropout:" + str(cf.rnn_dropout))
        info_logger("[Setting] rnn_num_layers:" + str(cf.rnn_num_layers))
        info_logger("[Setting] schedual_sampling_prob:" + str(cf.schedual_sampling_prob))
        info_logger("[Setting] gnn_concat:" + str(cf.gnn_concat))
        info_logger("[Setting] gnn_concat_uml:" + str(cf.gnn_concat_uml))
        info_logger("[Setting] uml path:" + str(cf.uml_path))

    if cf.modeltype == "h-deepcom" or cf.modeltype == "codenn":
        info_logger("[Setting] beam_search_method: %s" % cf.beam_search_method)
        info_logger("[Setting] beam_width: %d" % cf.beam_width)

    if not hasattr(cf, "random_seed"):
        cf.random_seed = 0    
    info_logger("[Setting] random_seed: " + str(cf.random_seed))


def read_uml_format_data(path, uml_path):
    data = read_pickle_data(path)  # dataset_uml.pkl
    # load uml data
    uml = torch.load(uml_path)   # uml_dataset.pt
    uml_train = uml["train"]
    try:
        uml_val = uml["valid"]
    except:
        uml_val = uml["val"]
    uml_test = uml["test"]
    
    train_m2u = data["m2utrain"]
    val_m2u = data["m2uval"]
    test_m2u = data["m2utest"]
    
    train_m2c = data["m2ctrain"]
    val_m2c = data["m2cval"]
    test_m2c = data["m2ctest"]
                   
    # load train, valid, test data and vocabulary
    train_summary = data["ctrain"]
    train_code = data["dtrain"]

    # valid info
    val_summary = data["cval"]
    val_code = data["dval"]
    val_ids = list(data["cval"].keys())

    # test info
    test_summary = data["ctest"]
    test_code = data["dtest"]
    test_ids = list(data['ctest'].keys())

    # vocabulary info
    summary_vocab = data["comstok"]
    code_vocab = data["datstok"]

    # i2w info
    summary_token_i2w = summary_vocab["i2w"]
    code_token_i2w = code_vocab["i2w"]

    summary_vocab_size = data["config"]["comvocabsize"]
    code_vocab_size = data["config"]["datvocabsize"]

    train_sbt = None
    val_sbt = None
    test_sbt = None
    sbt_vocab_size = -1
    if cf.enable_sbt:
        train_sbt = data["strain"]
        val_sbt = data["sval"]
        test_sbt = data["stest"]
        sbt_vocab_size = data["config"]["smlvocabsize"]
        
    train_func = None
    val_func = None
    test_func = None
    if cf.enable_func:
        train_func = data["ftrain"]
        val_func = data["fval"]
        test_func = data["ftest"]
        sbt_vocab_size = data["config"]["smlvocabsize"]

    summary_len = data["config"]["comlen"]

    cf.UNK_token_id = summary_vocab_size - 1

    # obtain DataLoader for iteration
    train_dataset = CodeSummaryUmlDataset(summary=train_summary, code=train_code,
                                          sbt=train_sbt, mapping={"method2uml": train_m2u, "method2class": train_m2c}, func_name = train_func)
    val_dataset = CodeSummaryUmlDataset(summary=val_summary, code=val_code,
                                        sbt=val_sbt, mapping={"method2uml": val_m2u, "method2class": val_m2c}, func_name = val_func)
    test_dataset = CodeSummaryUmlDataset(summary=test_summary, code=test_code,
                                         sbt=test_sbt, mapping={"method2uml": test_m2u, "method2class": test_m2c}, func_name = test_func)

    train_data_loader = DataLoader(train_dataset, shuffle=True, batch_size=cf.batch_size, pin_memory=True,
                                       num_workers=cf.num_subprocesses, drop_last=True)
    val_data_loader = DataLoader(val_dataset, shuffle=False, batch_size=cf.batch_size, pin_memory=True,
                                 num_workers=cf.num_subprocesses, drop_last=True)
    test_data_loader = DataLoader(test_dataset, shuffle=False, batch_size=cf.batch_size, pin_memory=True,
                                  num_workers=cf.num_subprocesses, drop_last=True)
    
    return train_dataset, val_dataset, test_dataset, train_data_loader, val_data_loader, test_data_loader, \
        code_vocab_size, sbt_vocab_size, summary_vocab_size, summary_vocab, summary_len, val_ids, test_ids, uml_train, uml_val, uml_test
                   

def main_uml():
    t0 = time.perf_counter()
    print("in_path")
    train_dataset, val_dataset, test_dataset, train_data_loader, val_data_loader, test_data_loader, code_vocab_size, \
        sbt_vocab_size, summary_vocab_size, summary_vocab, summary_len, val_ids, test_ids, uml_train, uml_val, uml_test = \
        read_uml_format_data(cf.in_path, cf.uml_path)
    summary_len = len(train_dataset.summary[0])
    # trgs_full_test, _ = load_summary(summary_len, cf.batch_size, test_ids, 'test', use_full_sum=True)
    # trgs_full_val, _ = load_summary(summary_len, cf.batch_size, val_ids, "valid", use_full_sum=True)  # full length summary
    trgs_trunc_test, trgs_test_id = load_summary(summary_len, cf.batch_size, test_ids, 'test', use_full_sum=cf.use_full_sum)  # truncated summary by sum_len
    try:
        trgs_trunc_val, trgs_val_id = load_summary(summary_len, cf.batch_size, val_ids, "valid", use_full_sum=cf.use_full_sum)  # truncated summary by sum_len
    except:
        trgs_trunc_val, trgs_val_id = load_summary(summary_len, cf.batch_size, val_ids, "val", use_full_sum=cf.use_full_sum)  # truncated summary by sum_len

    model = create_model(cf.modeltype, code_vocab_size, sbt_vocab_size, summary_vocab_size, summary_len, enable_sbt=cf.enable_sbt,
                         enable_uml=cf.enable_uml)
    debug_logger(torch_summarize(model))
    debug_logger('The model has %s trainable parameters' % str(count_parameters(model)))

    move_model_to_device(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cf.lr, weight_decay=cf.weight_decay)
    loss_fn = nn.CrossEntropyLoss(ignore_index=cf.PAD_token_id)

    t1 = time.perf_counter()
    info_logger("Finish Preparation %.2f secs [Total %.2f secs]" % (t1 - t0, t1 - t0))
    info_logger("code_vocab_size %d, sbt_vocab_size %d, summary_vocab_size %d" % (
        code_vocab_size, sbt_vocab_size, summary_vocab_size))
    info_logger("train %d, valid %d , test %d" % (len(train_dataset), len(val_dataset), len(test_dataset)))

    # uml_data = {"train": uml_train, "valid": uml_val, "test": uml_test}
    train_loss_list = []
    codeBert_bleus_val = {}
    codeBert_bleus_val[0] = -1
    best_epoch = 0
    best_val_bleu = 0
    for epoch in range(1, cf.num_epochs + 1):
        t2 = time.perf_counter()
        train_loss = train_uml(model, train_data_loader, optimizer, loss_fn, uml_train)
        train_loss_list.append(train_loss)
        t3 = time.perf_counter()

        info_logger("Epoch %d: Train Loss: %.3f, %.2f secs [Total %.2f secs]" % (epoch, train_loss, t3 - t2, t3 - t0))

        if epoch % cf.eval_frequency == 0:

            info_logger("---------use_full_sum = False: ------")
            ret_val, codeBert_bleu_val, codeBert_bleu_stem_val, _ = calculate_bleu_uml(model, val_data_loader,summary_vocab['i2w'], trgs_trunc_val, cf.trim_til_EOS, uml_val)
            codeBert_bleus_val[epoch] = codeBert_bleu_val
            info_logger(ret_val)

        if codeBert_bleus_val[epoch] > best_val_bleu:
            best_val_bleu = codeBert_bleus_val[epoch]
            best_epoch = epoch
            path = os.path.join("./model", cf.out_path)
            # path = os.path.join("./model", "model.pt")
            torch.save(model, path)

    info_logger("max codeBert_bleus_valid is %s" % str(codeBert_bleus_val[best_epoch]))
    info_logger("best epoch is %s" % str(best_epoch))
    model = load_model(path)
    info_logger("---------use_full_sum = False: ------")
    test_ret, codeBert_bleu_test, codeBert_bleu_stem_test, preds = calculate_bleu_uml(model, test_data_loader, summary_vocab['i2w'], trgs_trunc_test, cf.trim_til_EOS, uml_test)
    info_logger(test_ret)
    info_logger("max codeBert_bleus_test is %s" % str(codeBert_bleu_test))
    info_logger("max codeBert_bleus_stem_test is %s" % str(codeBert_bleu_stem_test))
    metetor_rouge_cider(trgs_trunc_test, preds)

    # info_logger("---------use_full_sum = True: ------")
    # test_ret_full, codeBert_bleu_test_full, codeBert_bleu_stem_test_full, preds = calculate_bleu_uml(model, test_data_loader,summary_vocab['i2w'],trgs_full_test,cf.trim_til_EOS, uml_test)
    # info_logger(test_ret_full)
    # info_logger("max codeBert_bleus_test is %s" % str(codeBert_bleu_test_full))
    # info_logger("max codeBert_bleus_stem_test is %s" % str(codeBert_bleu_stem_test_full))
    # metetor_rouge_cider(trgs_full_test, preds)
    info_logger("train loss is  %s" % str(train_loss_list))


def main_seq():
    t0 = time.perf_counter()

    train_dataset, val_dataset, test_dataset, train_data_loader, val_data_loader, test_data_loader, code_vocab_size, \
        sbt_vocab_size, summary_vocab_size, summary_vocab, summary_len, val_ids, test_ids = read_funcom_format_data(cf.in_path)
    #
    # trgs, fids = load_summary(summary_len, cf.batch_size, test_ids, 'test', use_full_sum=cf.use_full_sum)

    summary_len = len(train_dataset.summary[0])
    # trgs_full_test, _ = load_summary(summary_len, cf.batch_size, test_ids, 'test', use_full_sum=True)
    # trgs_full_val, _ = load_summary(summary_len, cf.batch_size, val_ids, "valid", use_full_sum=True)  # full length summary
    trgs_trunc_test, trgs_test_id = load_summary(summary_len, cf.batch_size, test_ids, 'test', use_full_sum= cf.use_full_sum)  # truncated summary by sum_len
    try:
        trgs_trunc_val, trgs_val_id = load_summary(summary_len, cf.batch_size, val_ids, "valid", use_full_sum=cf.use_full_sum)  # truncated summary by sum_len
    except:
        trgs_trunc_val, trgs_val_id = load_summary(summary_len, cf.batch_size, val_ids, "val", use_full_sum=cf.use_full_sum)  # truncated summary by sum_len
    model = create_model(cf.modeltype, code_vocab_size, sbt_vocab_size, summary_vocab_size, summary_len, enable_sbt=cf.enable_sbt,
                         enable_uml=cf.enable_uml)

    debug_logger(torch_summarize(model))
    debug_logger('The model has %s trainable parameters' % str(count_parameters(model)))

    move_model_to_device(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=cf.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=cf.PAD_token_id)

    t1 = time.perf_counter()
    info_logger("Finish Preparation %.2f secs [Total %.2f secs]" % (t1 - t0, t1 - t0))
    info_logger("code_vocab_size %d, sbt_vocab_size %d, summary_vocab_size %d" % (
        code_vocab_size, sbt_vocab_size, summary_vocab_size))
    info_logger("train %d, val %d, test %d" % (len(train_dataset), len(val_dataset), len(test_dataset)))

    train_loss_list = []
    codeBert_bleus_val = {}
    codeBert_bleus_val[0] = -1
    best_epoch = 0
    best_val_bleu = 0
    for epoch in range(1, cf.num_epochs + 1):
        t2 = time.perf_counter()
        train_loss = train_seq(model, train_data_loader, optimizer, loss_fn)
        train_loss_list.append(train_loss)
        t3 = time.perf_counter()
        info_logger("Epoch %d: Train Loss: %.3f, %.2f secs [Total %.2f secs]" % (epoch, train_loss, t3 - t2, t3 - t0))

        if epoch % cf.eval_frequency == 0:
            info_logger("---------use_full_sum = False: ------")
            # ret_val, codeBert_bleu_val, codeBert_bleu_stem_val, _ = calculate_bleu(model, test_data_loader, summary_vocab['i2w'], trgs_trunc_val, cf.trim_til_EOS)
            ret_val, codeBert_bleu_val, codeBert_bleu_stem_val, _ = calculate_bleu(model, val_data_loader, summary_vocab['i2w'], trgs_trunc_val, cf.trim_til_EOS)
            codeBert_bleus_val[epoch] = codeBert_bleu_val
            info_logger(ret_val)

            # info_logger("---------use_full_sum = True: ------")
            # ret_val_full, _, _, _ = calculate_bleu(model, val_data_loader, summary_vocab['i2w'],trgs_full_val,cf.trim_til_EOS)
            # info_logger(ret_val_full)

        if codeBert_bleus_val[epoch] > best_val_bleu:
            best_val_bleu = codeBert_bleus_val[epoch]
            best_epoch = epoch
            path = os.path.join("./model",  cf.out_path)
            torch.save(model, path)


    info_logger("max codeBert_bleus_valid is %s" % str(codeBert_bleus_val[best_epoch]))
    info_logger("best epoch is %s" % str(best_epoch))
    model = load_model(path)
    info_logger("---------use_full_sum = False: ------")
    test_ret, codeBert_bleu_test, codeBert_bleu_stem_test, preds = calculate_bleu(model, test_data_loader, summary_vocab['i2w'], trgs_trunc_test, cf.trim_til_EOS)
    info_logger(test_ret)
    info_logger("max codeBert_bleus_test is %s" % str(codeBert_bleu_test))
    info_logger("max codeBert_bleus_stem_test is %s" % str(codeBert_bleu_stem_test))
    metetor_rouge_cider(trgs_trunc_test, preds)

    # info_logger("---------use_full_sum = True: ------")
    # test_ret_full, codeBert_bleu_test_full, codeBert_bleu_stem_test_full, preds = calculate_bleu(model, test_data_loader,summary_vocab['i2w'],trgs_full_test, cf.trim_til_EOS)
    # info_logger(test_ret_full)
    # info_logger("max codeBert_bleus_test is %s" % str(codeBert_bleu_test_full))
    # info_logger("max codeBert_bleus_stem_test is %s" % str(codeBert_bleu_stem_test_full))
    # metetor_rouge_cider(trgs_full_test, preds)
    info_logger("train loss is  %s" % str(train_loss_list))


def show_command(cf):
    print("simplified command: python runGruUml.py -modeltype=%s -root=%s -data=%s -docstring_tokens_split_path=%s -m2u_m2c_path=%s \
        -uml_path=%s -source_code_path=%s -multi_gpu=%d -gpu_count=%d -batch_size=%d" %
          (cf.modeltype, args.root, args.data, args.docstring_tokens_split_path, args.m2u_m2c_path, args.uml_path,
           args.source_code_path, cf.multi_gpu, cf.gpu_count, cf.batch_size))
    print("full command: python runGruUml.py -modeltype=%s -root=%s -data=%s -docstring_tokens_split_path=%s -m2u_m2c_path=%s \
            -uml_path=%s -source_code_path=%s -multi_gpu=%d -gpu_count=%d -use_oov_sum=%s \
          -trim_til_EOS=%s -EXP=%s -DEBUG=%s -beam_search_method=%s -beam_width=%d -batch_size=%d -random_seed=%d \
          -code_dim=%d -summary_dim=%d -sbt_dim=%d -rnn_hidden_size=%d -lr=%f -num_epochs=%d -num_subprocesses=%d \
          -eval_frequency=%d" %
          (cf.modeltype, args.root, args.data, args.docstring_tokens_split_path, args.m2u_m2c_path, args.uml_path,
           args.source_code_path, cf.multi_gpu, cf.gpu_count, cf.use_oov_sum, cf.trim_til_EOS, cf.EXP, cf.DEBUG, cf.beam_search_method,
           cf.beam_width, cf.batch_size, cf.random_seed, cf.code_dim, cf.summary_dim, cf.sbt_dim, cf.rnn_hidden_size, cf.lr, cf.num_epochs, cf.num_subprocesses,
           cf.eval_frequency))


def main():
    make_directory("model")
    
    set_config()
    # show_command(cf)
    cf.DEBUG = False
    set_logger(cf.DEBUG)
    basic_info_logger()
    set_device(cf.gpu_id)
    set_seed(cf.random_seed)
    if 'uml' in cf.modeltype:
        main_uml()
    else:
        main_seq()


if __name__ == "__main__":
    main()
