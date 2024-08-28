#!/usr/bin/env python
# !-*-coding:utf-8 -*-
import random
import torch
import torch.nn as nn
import os

from loadModel import load_summary
from models.ASTAttGRU import AstAttGRUModel
from models.AttGRU import AttGRUModel
from models.HDeepCom import HDeepComModel
from models.codenn import CodeNNModel
from util.DataUtil import read_pickle_data, make_directory, read_funcom_format_data, get_file_name,\
    set_config_data_processing, str_to_bool
from util.EvaluateUtil import calculate_bleu, compute_predictions
from util.GPUUtil import move_model_to_device, move_to_device, np
from util.LoggerUtil import info_logger, set_logger, debug_logger, torch_summarize, count_parameters
from util.Config import Config as cf
import time
from util.GPUUtil import set_device

import argparse
from torch.backends import cudnn
import time


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False  # if benchmark=True, deterministic will be Fals
    cudnn.deterministic = True


def train(model, train_data_loader, optimizer, loss_fn):
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
        elif cf.modeltype == 'att-gru':
            output = model(method_code=train_batch_data[0], method_summary=train_batch_data[1], use_teacher=True)[-1]
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


# def evaluate(model, test_data_loader, loss_fn):
#     model.eval()
#
#     cumulative_loss = 0
#     step_size = 0
#
#     for test_batch_data in test_data_loader:
#
#         for i in range(len(test_batch_data)):
#             test_batch_data[i] = move_to_device(test_batch_data[i])
#
#         with torch.no_grad():
#             if cf.modeltype == 'ast-att-gru':
#                 output = model(method_code=test_batch_data[0], method_sbt=test_batch_data[1],
#                                method_summary=test_batch_data[2], use_teacher=False)[-1]
#             elif cf.modeltype == 'att-gru':
#                 output = model(method_code=test_batch_data[0], method_summary=test_batch_data[1], use_teacher=False)[-1]
#             else:
#                 raise Exception("Unrecognized Model: ", str(cf.modeltype))
#
#             sum_vocab_size = output.shape[-1]
#             # output: batch_size, summary_length, sum_vocab_size
#             output = output.view(-1, sum_vocab_size)
#             # output: batch_size * summary_length, sum_vocab_size
#             trg = test_batch_data[-1].view(-1)
#             # trg = [batch size * summary_length]
#
#             loss = loss_fn(output, trg)
#
#             cumulative_loss += loss.item()
#             step_size += 1
#
#     return cumulative_loss / step_size


def create_model(model_type, token_vocab_size, sbt_vocab_size, summary_vocab_size, summary_len):
    if model_type == 'att-gru':
        # basic attention GRU model based on Nematus architecture
        return AttGRUModel(token_vocab_size, summary_vocab_size)
    elif model_type == 'ast-att-gru':
        # attention GRU model with added AST information from srcml.
        return AstAttGRUModel(token_vocab_size, sbt_vocab_size, summary_vocab_size)
    elif model_type == 'codenn':
        # attention LSTM model, refer to ACL 2016 paper.
        return CodeNNModel(token_vocab_size, summary_vocab_size, summary_len)
    elif model_type == 'h-deepcom':
        return HDeepComModel(token_vocab_size, sbt_vocab_size, summary_vocab_size, summary_len)
    else:
        raise Exception("Unrecognized Model: ", str(model_type))


def set_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu_id', required=False)
    parser.add_argument('-data', required=False)
    # parser.add_argument('-dataset_path', type=str, required=False)

    parser.add_argument('-batch_size', required=False)
    parser.add_argument('-modeltype', required=False)

    parser.add_argument('-code_dim', required=False)
    parser.add_argument('-summary_dim', required=False)
    parser.add_argument('-sbt_dim', required=False)

    parser.add_argument('-rnn_hidden_size', required=False)

    parser.add_argument('-lr', required=False)

    parser.add_argument('-epoch', required=False)

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
    parser.add_argument('-dvoc', "--code_vocab_size",required=False)
    parser.add_argument('-cvoc', "--summary_vocab_size", required=False)
    parser.add_argument('-svoc', "--sbt_vocab_size", required=False)

    # dataset
    parser.add_argument('-dataset_path', type=str, required=False)

    # beam search
    parser.add_argument('-beam_search_method', required=False)
    parser.add_argument('-beam_width', required=False)

    args = parser.parse_args()

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
    if args.dataset_path:
        cf.dataset_path = args.dataset_path

    if args.gpu_id:
        cf.gpu_id = int(args.gpu_id)
    if args.data:
        cf.in_path = args.data
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
    if args.epoch:
        cf.num_epochs = int(args.epoch)

    filename = get_file_name()
    if args.dataset_path:
        cf.dataset_path = args.dataset_path

        cf.in_path = os.path.join(cf.dataset_path, filename)

    cf.out_path = filename + "_mt" + str(cf.modeltype) + "_bs" + str(cf.batch_size) + \
                  "_ddim" + str(cf.code_dim) + "_cdim" + str(cf.summary_dim) + "_sdim" + str(cf.sbt_dim) +\
                  "_hdim" + str( cf.rnn_hidden_size) + "_lr" + str( cf.lr) + "_" + time.strftime("%Y%m%d%H%M%S")
    if args.out_path:
        cf.out_path = args.out_path

    if args.beam_search_method:
        cf.beam_search_method = args.beam_search_method

    if args.beam_width:
        cf.beam_width = int(args.beam_width)

def basic_info_logger():
    info_logger("[Setting] EXP: %s" % (str(cf.EXP)))
    info_logger("[Setting] DEBUG: %s" % (str(cf.DEBUG)))
    # info_logger("[Setting] trimTilEOS: %s" % (str(cf.trimTilEOS)))
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
    info_logger("[Setting] num_epochs: %d" % cf.num_epochs)
    info_logger("[Setting] num_subprocesses: %d" % cf.num_subprocesses)
    info_logger("[Setting] eval_frequency: %d" % cf.eval_frequency)
    if cf.out_path != "":
        info_logger("[Setting] out_path: %s" % cf.out_path)

    if cf.modeltype == "h-deepcom" or cf.modeltype == "codenn":
        info_logger("[Setting] beam_search_method: %s" % cf.beam_search_method)
        info_logger("[Setting] beam_width: %d" % cf.beam_width)


def main():
    set_seed(123)
    set_config()
    set_logger(cf.DEBUG)
    basic_info_logger()
    set_device(cf.gpu_id)
    t0 = time.perf_counter()

    train_dataset, val_dataset, test_dataset, train_data_loader, val_data_loader, test_data_loader, code_vocab_size, \
        sbt_vocab_size, summary_vocab_size, summary_vocab, summary_len, val_ids, test_ids = read_funcom_format_data(cf.in_path)
    summary_len = len(train_dataset.summary[0])
    trgs, fids= load_summary(summary_len, cf.batch_size, test_ids, 'test', use_full_sum=cf.use_full_sum)

    # model = create_model(cf.modeltype, code_vocab_size, sbt_vocab_size, summary_vocab_size, enable_sbt=cf.enable_sbt,
    #                      enable_uml=cf.enable_uml)
    model = create_model(cf.modeltype, code_vocab_size, sbt_vocab_size, summary_vocab_size, summary_len)

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

    for epoch in range(1, cf.num_epochs + 1):
        t2 = time.perf_counter()
        train_loss = train(model, train_data_loader, optimizer, loss_fn)

        t3 = time.perf_counter()

        info_logger("Epoch %d: Train Loss: %.3f, %.2f secs [Total %.2f secs]" % (epoch, train_loss, t3 - t2, t3 - t0))

        if epoch % cf.eval_frequency == 0:
            info_logger(calculate_bleu(model, test_data_loader, summary_vocab['i2w'], trgs, trimTilEOS=cf.trimTilEOS)[0])

    # ToDo: save the best epoch
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training
        if cf.out_path != "":
            make_directory("./model")
            path = os.path.join("./model", str(epoch)+cf.out_path)
            info_logger("Output Model to %s" % path)
            torch.save(model, path)


if __name__ == "__main__":
    main()
