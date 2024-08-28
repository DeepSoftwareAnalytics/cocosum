import argparse
import math
import time
import torch
from runGruUml import read_uml_format_data
from util.Config import Config as cf
from util.DataUtil import read_pickle_data, read_funcom_format_data, save_pickle_data
from util.EvaluateUtil import compute_predictions, compute_predictions_uml
from util.GPUUtil import set_device, move_model_to_device
from util.cider.cider import Cider
from util.meteor.meteor import Meteor
from util.rouge.rouge import Rouge


def metetor_rouge_cider(refs, preds):
    refs_dict = {}
    preds_dict = {}
    for i in range(len(preds)):
        preds_dict[i] = [" ".join(preds[i])]
        refs_dict[i] = [" ".join(refs[i][0])]

    score_Rouge, scores_Rouge = Rouge().compute_score(refs_dict, preds_dict)
    print("ROUGe: ", score_Rouge)

    score_Cider, scores_Cider = Cider().compute_score(refs_dict, preds_dict)
    print("Cider: ", score_Cider)

    score_Meteor, scores_Meteor = Meteor().compute_score(refs_dict, preds_dict)
    print("Meteor: ", score_Meteor)


def load_summary(sum_len, batch_size, ids, col, use_full_sum):
    """"
    col: 'train', 'val', 'test'
    """
    docstring_tokens_split_path = cf.docstring_tokens_split_path
    sum_gt = read_pickle_data(docstring_tokens_split_path)[col]  # summary ground truth of validation set
    new_len = batch_size * (math.floor(len(ids) / batch_size))
    if use_full_sum:
        trgs = [sum_gt[i] for i in ids[: new_len]]   # use full ground truth summary
    else:
        trgs = [sum_gt[i][:sum_len - 1] for i in ids[: new_len]]   # use the first sum_len ground truth summary
    trgs = [[t] for t in trgs]  # a list of lists
    return trgs, ids[: new_len]


def load_model(model_path):
    if cf.multi_gpu:
        torch.distributed.init_process_group(backend="nccl")
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        set_device(local_rank)
        model = torch.load(model_path)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
                                                          find_unused_parameters=True)
        return model
    else:
        set_device(cf.gpu_id)
        model = torch.load(model_path, map_location=torch.device('cpu'))
        move_model_to_device(model)
        return model


def predict_result( model_path,output_prediction_path,output_prediction_filename):
    model = load_model(model_path)
    set_device(cf.gpu_id)
    # move_model_to_device(model)
    train_dataset, val_dataset, test_dataset, train_data_loader, val_data_loader, test_data_loader, code_vocab_size, \
    sbt_vocab_size, summary_vocab_size, summary_vocab, summary_len, val_ids, test_ids = read_funcom_format_data(cf.in_path)

    cf.use_full_sum = False
    summary_len = len(train_dataset.summary[0])
    trgs, test_fids = load_summary(summary_len, cf.batch_size, test_ids, 'test', use_full_sum=cf.use_full_sum)
    preds =compute_predictions(model, test_data_loader, summary_vocab['i2w'],  cf.trim_til_EOS)
    d = {" preds": preds, "trgs":trgs, "test_fids":test_fids}
    save_pickle_data(output_prediction_path, output_prediction_filename, d)
    return preds,trgs, test_fids

def predict_uml_result( model_path,output_prediction_path,output_prediction_filename):
    model = load_model(model_path)
    set_device(cf.gpu_id)
    # move_model_to_device(model)
    train_dataset, val_dataset, test_dataset, train_data_loader, val_data_loader, test_data_loader, code_vocab_size, \
    sbt_vocab_size, summary_vocab_size, summary_vocab, summary_len, val_ids, test_ids, uml_train, uml_val, uml_test = \
        read_uml_format_data(cf.in_path, cf.uml_path)

    cf.use_full_sum = False
    summary_len = len(train_dataset.summary[0])
    trgs, test_fids = load_summary(summary_len, cf.batch_size, test_ids, 'test', use_full_sum=cf.use_full_sum)
    preds =compute_predictions_uml(model, test_data_loader, summary_vocab['i2w'],  cf.trim_til_EOS, uml_test)
    d = {" preds": preds, "trgs":trgs, "test_fids":test_fids}
    save_pickle_data(output_prediction_path, output_prediction_filename, d)
    return preds,trgs, test_fids


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('-modeltype', required=True)
    parser.add_argument('-model_file', required=True)
    parser.add_argument('-model_root_path', required=True)
    parser.add_argument('-prediction_path', required=True)
    parser.add_argument('-multi_gpu', action='store_true')
    args = parser.parse_args()

    cf.modeltype = args.modeltype  # ast-att-gru, h-deepcom, codenn, att-gru
    if args.multi_gpu:
        cf.multi_gpu = True
    else:
        cf.multi_gpu = False
    
    model_file = args.model_file
    model_path = args.model_root_path + model_file
    output_prediction_file = model_file + '.pred'
    prediction_path = args.prediction_path

    st = time.perf_counter()
    if "uml" in cf.modeltype:
        preds,trgs, test_fids = predict_uml_result(model_path,prediction_path,output_prediction_file)
    else:
        preds,trgs, test_fids = predict_result(model_path,prediction_path,output_prediction_file)
    print("time cost: ",time.perf_counter() -st)

    metetor_rouge_cider(trgs, preds)
