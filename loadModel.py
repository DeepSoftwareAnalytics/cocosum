import math
import torch
from util.Config import Config as cf
from util.DataUtil import read_pickle_data, read_funcom_format_data
from util.EvaluateUtil import calculate_bleu
from util.GPUUtil import set_device, move_model_to_device


def load_summary(sum_len, batch_size, ids, col, use_full_sum):
    """"
    col: 'train', 'val', 'test'
    """
    docstring_tokens_split_path = cf.docstring_tokens_split_path
    # docstring_tokens_split_path = "/mnt/Data/csn/summary/cfp1_csi1_cfd0_clc1.pkl"
    sum_gt = read_pickle_data(docstring_tokens_split_path)[col]  # summary ground truth of validation set
    new_len = batch_size * (math.floor(len(ids) / batch_size))
    if use_full_sum:
        trgs = [sum_gt[i] for i in ids[: new_len]]   # use full ground truth summary
    else:
        trgs = [sum_gt[i][:sum_len - 1] for i in ids[: new_len]]   # use the first sum_len ground truth summary
    trgs = [[t] for t in trgs]  # a list of lists
    return trgs, ids[: new_len]


def load_model(model_path):
    set_device(cf.gpu_id)
    model = torch.load(model_path)
    move_model_to_device(model)
    return model


def main():
    # model_path = "output/model2.pth"
    # output_prediction_path = "output/model2_pred_results.txt"
    # model_path = "output/csn_200219/data_csn_voc1000_d100_c13_128_tdim_128_cdim_128_rdim_256_lr_0.001"
    # output_prediction_path = "output/csn_200219/data_csn_voc1000_d100_c13_128_tdim_128_cdim_128_rdim_256_lr_0.001.pred"
    # model_path = "/mnt/enshi/CodeSum316/output_code_filter_voc1w_diffdc/csn_code_filter_d50c10s100_voc1w-att-gru-256-128-128-0-128-0.001-20200317202802.pth"
    # output_prediction_path = "/mnt/enshi/CodeSum316/output_code_filter_voc1w_diffdc/csn_code_filter_d50c10s100_voc1w-att-gru-256-128-128-0-128-0.001-20200317202802.prd"
    # model_path = './model/1csn_code_filter_d80c16s100_vocd10000c10000s10000.pkl'
    model_path = './model/1test.pkl'
    # output_prediction_path = './model/1csn_code_filter_d80c16s100_vocd10000c10000s10000.prd'
    output_prediction_path = './model/1test.prd'
    model = load_model(model_path)
    # set_device(cf.gpu_id)
    # move_model_to_device(model)

    train_dataset, val_dataset, test_dataset, train_data_loader, val_data_loader, test_data_loader, code_vocab_size, \
    sbt_vocab_size, summary_vocab_size, summary_vocab, summary_len, val_ids, test_ids = read_funcom_format_data(cf.in_path)

    summary_len = len(train_dataset.summary[0])
    # preds = compute_predictions(model, test_data_loader, summary_token_i2w, trimTilEOS=cf.trimTilEOS)
    trgs, test_fids = load_summary(summary_len, cf.batch_size, test_ids, 'test', use_full_sum=cf.use_full_sum)
    # bleu_results = calc_all_metrics(preds=preds, refs=trgs)

    bleu_results, preds = calculate_bleu(model, test_data_loader, summary_vocab['i2w'], trgs, trimTilEOS=cf.trim_til_EOS)

    print(bleu_results[0])
    print("Output prediction results to", output_prediction_path)

    output_results = [str(bleu_results)]
    for i in range(len(preds)):
        output_results.append("fid: " + str(test_fids[i]))
        output_results.append(" ".join(preds[i]))
        output_results.append(" ".join(trgs[i][0]))
        output_results.append("---------------------------------------")

    with open(output_prediction_path, "w") as output:
        for line in output_results:
            output.write(line + "\r")


if __name__ == "__main__":
    main()
