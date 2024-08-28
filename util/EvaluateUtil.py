import nltk
import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from torch.utils.data.dataloader import DataLoader
# import sys
# sys.path.append("./")
from util.Config import Config as cf
from util.CustomedBleu.bleu import _bleu
from util.CustomedBleu.smooth_bleu import smooth_bleu
from util.DataUtil import read_pickle_data,stem_and_replace_predict_target
from util.Dataset import CodeSummaryDataset
from util.GPUUtil import move_to_device, move_pyg_to_device
from torch_geometric.data import Batch
# from rouge import Rouge
from util.meteor.meteor import Meteor
from util.rouge.rouge import Rouge
from util.cider.cider import Cider


class BeamSearchNode(object):

    def __init__(self, hidden_state, previous_node, token_id, cumulative_prob, length, next_prob_distribution):
        # tensor of size hidden_size or tuple of two hidden states with size hidden_size
        self.hidden_state = hidden_state
        # reference to previous node
        self.previous_node = previous_node
        # tensor of size 1
        self.token_id = token_id
        # scalar
        self.cumulative_prob = cumulative_prob
        # scalar
        self.length = length
        # tensor of size summary_voc_size: prob distribution for NEXT prediction
        self.next_prob_distribution = next_prob_distribution

    def __lt__(self, other):
        return self.cumulative_prob < other.cumulative_prob


# Leclair approch. There will be <UNK> in test summary ground truth!
def calculate_bleu_old(model, test_data_loader):
    model.eval()

    preds = []
    trgs = []

    for test_batch_data in test_data_loader:
        trg = test_batch_data[-1]
        for i in range(len(test_batch_data)):
            test_batch_data[i] = move_to_device(test_batch_data[i])

        with torch.no_grad():
            if cf.modeltype == 'ast-att-gru':
                output = model(method_code=test_batch_data[0], method_sbt=test_batch_data[1],
                               method_summary=test_batch_data[2], use_teacher=False)[-1]
            elif cf.modeltype == 'att-gru' or cf.modeltype == 'codenn':
                output = model(method_code=test_batch_data[0], method_summary=test_batch_data[1], use_teacher=False)[-1]
            else:
                raise Exception("Unrecognized Model: ", str(cf.modeltype))

            # output: batch_size, summary_length - 1, sum_vocab_size
            # pred: batch_size, summary_length - 1
            pred = torch.argmax(output, dim=2)

        # https://github.com/mcmillco/funcom/blob/41c737903a286e428493705e53c9c0e5946d2af6/bleu.py#L92
        # https://github.com/mcmillco/funcom/blob/41c737903a286e428493705e53c9c0e5946d2af6/bleu.py#L107
        # If p or t contain <s>, </s>, <UNK>, <NULL>,
        # they will be included as output.
        for i in range(cf.batch_size):
            p = pred[i].cpu().tolist()
            p = [value for value in p if
                 value not in [cf.PAD_token_id, cf.SOS_token_id, cf.EOS_token_id, cf.UNK_token_id]]
            preds.append(p)

            t = trg[i].cpu().tolist()
            t = [value for value in t if
                 value not in [cf.PAD_token_id, cf.SOS_token_id, cf.EOS_token_id, cf.UNK_token_id]]

            trgs.append([t])

    return bleu_so_far(trgs, preds)


def calculate_bleu_pretrain(model, test_data_loader, summary_i2w, trgs, trimTilEOS):
    preds = compute_predictions_pretrain(model, test_data_loader, summary_i2w, trimTilEOS)
    # return calc_all_metrics(preds, trgs), preds
    return calc_all_metrics(preds, trgs)


def compute_predictions_pretrain(model, test_data_loader, summary_i2w, trimTilEOS):
    model.eval()

    preds = []
    for test_batch_data in test_data_loader:
        for i in range(len(test_batch_data)):
            if cf.multi_gpu==True:
                test_batch_data[i] = test_batch_data[i].to("cuda")
            else:
                test_batch_data[i] = move_to_device(test_batch_data[i])

        with torch.no_grad():
            if cf.modeltype == 'ast-att-gru-codebert':
                output = model(method_code=test_batch_data[0],source_mask=test_batch_data[1], method_summary=test_batch_data[2],
                                method_sbt=test_batch_data[3], use_teacher=False)[-1]
            # elif cf.modeltype == 'att-gru':
            #     output = model(method_code=test_batch_data[0], method_summary=test_batch_data[1], use_teacher=False)[-1]
            # elif cf.modeltype == 'codenn':
            #     output = model(method_code=test_batch_data[0], beam_width=cf.beam_width, is_test=True)[-1]
            # elif cf.modeltype == 'h-deepcom':
            #     output = model(method_code=test_batch_data[0], method_sbt=test_batch_data[1], beam_width=cf.beam_width,
            #                    is_test=True)[-1]
            # # elif cf.modeltype == "uml":
            #     output = model(
            else:
                raise Exception("Unrecognized Model: ", str(cf.modeltype))

            # output: batch_size, summary_length - 1, sum_vocab_size
            # pred: batch_size, summary_length - 1
            pred = torch.argmax(output, dim=2)

        # https://github.com/mcmillco/funcom/blob/41c737903a286e428493705e53c9c0e5946d2af6/bleu.py#L92
        # https://github.com/mcmillco/funcom/blob/41c737903a286e428493705e53c9c0e5946d2af6/bleu.py#L107
        # If p or t contain <s>, </s>, <UNK>, <NULL>,
        # they will be included as output.
        for i in range(cf.batch_size):
            p = pred[i].cpu().tolist()
            if trimTilEOS:
                if cf.EOS_token_id in p:  # truncate at first </s>
                    p = p[:p.index(cf.EOS_token_id) + 1]

            p = [summary_i2w[value] for value in p if
                    value not in [cf.PAD_token_id, cf.SOS_token_id, cf.EOS_token_id, cf.UNK_token_id]]

            preds.append(p)
            # t = trg[i].cpu().tolist()
            # t = [value for value in t if
            #      value not in [cf.PAD_token_id, cf.SOS_token_id, cf.EOS_token_id, cf.UNK_token_id]]
            # trgs.append([t])
    model.train()
    return preds


def calculate_bleu(model, test_data_loader, summary_i2w, trgs, trimTilEOS):
    preds = compute_predictions(model, test_data_loader, summary_i2w, trimTilEOS)
    # return calc_all_metrics(preds, trgs), preds
    return calc_all_metrics(preds, trgs)


def compute_predictions(model, test_data_loader, summary_i2w, trimTilEOS):
    model.eval()

    preds = []
    for test_batch_data in test_data_loader:
        # trg = test_batch_data[-1]
        for i in range(len(test_batch_data)):
            test_batch_data[i] = move_to_device(test_batch_data[i])

        with torch.no_grad():
            if cf.modeltype == 'ast-att-gru':
                output = model(method_code=test_batch_data[0], method_sbt=test_batch_data[1],
                               method_summary=test_batch_data[2], use_teacher=False)[-1]
            elif cf.modeltype == 'ast-att-transformer':
                output = model(method_code=test_batch_data[0], method_sbt=test_batch_data[1],
                               method_summary=test_batch_data[2], use_teacher=False)[-1]
            elif cf.modeltype == 'att-gru':
                output = model(method_code=test_batch_data[0], method_summary=test_batch_data[2], use_teacher=False)[-1]
            elif cf.modeltype == 'codenn':
                output = model(method_code=test_batch_data[0], beam_width=cf.beam_width, is_test=True)[-1]
            elif cf.modeltype == 'h-deepcom':
                output = model(method_code=test_batch_data[0], method_sbt=test_batch_data[1], beam_width=cf.beam_width,
                               is_test=True)[-1]
            # elif cf.modeltype == "uml":
            #     output = model(
            else:
                raise Exception("Unrecognized Model: ", str(cf.modeltype))

            # output: batch_size, summary_length - 1, sum_vocab_size
            # pred: batch_size, summary_length - 1
            pred = torch.argmax(output, dim=2)

        # https://github.com/mcmillco/funcom/blob/41c737903a286e428493705e53c9c0e5946d2af6/bleu.py#L92
        # https://github.com/mcmillco/funcom/blob/41c737903a286e428493705e53c9c0e5946d2af6/bleu.py#L107
        # If p or t contain <s>, </s>, <UNK>, <NULL>,
        # they will be included as output.
        for i in range(cf.batch_size):
            p = pred[i].cpu().tolist()
            if trimTilEOS:
                if cf.EOS_token_id in p:  # truncate at first </s>
                    p = p[:p.index(cf.EOS_token_id) + 1]

            p = [summary_i2w[value] for value in p if
                    value not in [cf.PAD_token_id, cf.SOS_token_id, cf.EOS_token_id, cf.UNK_token_id]]

            preds.append(p)
            # t = trg[i].cpu().tolist()
            # t = [value for value in t if
            #      value not in [cf.PAD_token_id, cf.SOS_token_id, cf.EOS_token_id, cf.UNK_token_id]]
            # trgs.append([t])
    model.train()
    return preds



def calculate_bleu_uml_pretrain(model, test_data_loader, summary_i2w, trgs, trimTilEOS, uml_data):
    preds = compute_predictions_uml_pretrain(model, test_data_loader, summary_i2w, trimTilEOS, uml_data)
    return calc_all_metrics(preds, trgs)


def compute_predictions_uml_pretrain(model, test_data_loader, summary_i2w, trimTilEOS, uml_data):

    model.eval()

    preds = []
    for test_batch_data in test_data_loader:
        for i in range(len(test_batch_data) - 2):
            test_batch_data[i] = move_to_device(test_batch_data[i])

        m2u = test_batch_data[-2]
        m2c = test_batch_data[-1]

        uml_dict = {}
        uml_list = []
        class_st_idx = [0]
        cnt = 0
        uml_index = []

        for it in m2u:
            # print("it",it)
            i = int(it)
            if it not in uml_dict:
                uml_dict[i] = cnt
                cnt += 1
                uml_list.append(uml_data[i])
                class_st_idx.append(class_st_idx[-1] + uml_data[i].num_nodes)
            uml_index.append(uml_dict[i])

        uml_index = torch.LongTensor(uml_index)

        uml_batch = Batch.from_data_list(uml_list)
        class_index = [class_st_idx[uml_dict[int(m2u[idx])]] + it for idx, it in enumerate(m2c)]
        class_index = torch.LongTensor(class_index)

        uml_index = move_to_device(uml_index)
        uml_batch = move_pyg_to_device(uml_batch)
        class_index = move_to_device(class_index)

        with torch.no_grad():
            if cf.enable_sbt:
                output = model(method_code=test_batch_data[0], source_mask=test_batch_data[1], method_sbt=test_batch_data[2],
                               method_summary=test_batch_data[3], use_teacher=False,
                               uml_data = uml_batch, class_index = class_index, uml_index = uml_index)[-1]
            else:
                output = model(method_code=test_batch_data[0], source_mask=test_batch_data[1], method_summary=test_batch_data[2], use_teacher=False,
                               uml_data = uml_batch, class_index = class_index, uml_index = uml_index)[-1]

            # output: batch_size, summary_length - 1, sum_vocab_size
            # pred: batch_size, summary_length - 1
            pred = torch.argmax(output, dim=2)

        # https://github.com/mcmillco/funcom/blob/41c737903a286e428493705e53c9c0e5946d2af6/bleu.py#L92
        # https://github.com/mcmillco/funcom/blob/41c737903a286e428493705e53c9c0e5946d2af6/bleu.py#L107
        # If p or t contain <s>, </s>, <UNK>, <NULL>,
        # they will be included as output.
        for i in range(cf.batch_size):
            p = pred[i].cpu().tolist()
            if trimTilEOS:
                if cf.EOS_token_id in p:  # truncate at first </s>
                    p = p[:p.index(cf.EOS_token_id) + 1]

            p = [summary_i2w[value] for value in p if
                    value not in [cf.PAD_token_id, cf.SOS_token_id, cf.EOS_token_id, cf.UNK_token_id]]

            preds.append(p)
            # t = trg[i].cpu().tolist()
            # t = [value for value in t if
            #      value not in [cf.PAD_token_id, cf.SOS_token_id, cf.EOS_token_id, cf.UNK_token_id]]
            # trgs.append([t])
    model.train()
    return preds


def calculate_bleu_uml(model, test_data_loader, summary_i2w, trgs, trimTilEOS, uml_data):
    # preds[0] ['this', 'the', 'the', 'for', 'the', 'the', 'the', 'if']
    # target[0][0] ['this', 'the', 'the', 'for', 'the', 'the', 'the', 'if']
    preds = compute_predictions_uml(model, test_data_loader, summary_i2w, trimTilEOS, uml_data)

    return calc_all_metrics(preds, trgs)


def compute_predictions_uml(model, test_data_loader, summary_i2w, trimTilEOS, uml_data):
    model.eval()

    preds = []
    for test_batch_data in test_data_loader:
        # trg = test_batch_data[-1]
        for i in range(len(test_batch_data) - 2):
            test_batch_data[i] = move_to_device(test_batch_data[i])

        m2u = test_batch_data[-2]
        m2c = test_batch_data[-1]

        uml_dict = {}
        uml_list = []
        class_st_idx = [0]
        cnt = 0
        uml_index = []

        for it in m2u:
            i = int(it)
            if it not in uml_dict:
                uml_dict[i] = cnt
                cnt += 1
                uml_list.append(uml_data[i])
                class_st_idx.append(class_st_idx[-1] + uml_data[i].num_nodes)
            uml_index.append(uml_dict[i])

        uml_index = torch.LongTensor(uml_index)

        uml_batch = Batch.from_data_list(uml_list)
        class_index = [class_st_idx[uml_dict[int(m2u[idx])]] + it for idx, it in enumerate(m2c)]
        class_index = torch.LongTensor(class_index)

        uml_index = move_to_device(uml_index)
        uml_batch = move_pyg_to_device(uml_batch)
        class_index = move_to_device(class_index)

        with torch.no_grad():
            if cf.modeltype == 'uml-transformer':
                output = model(method_code=test_batch_data[0], method_sbt=test_batch_data[1],
                               method_summary=test_batch_data[2], use_teacher=False,
                               uml_data=uml_batch, class_index=class_index, uml_index=uml_index)[-1]
            elif cf.modeltype == 'uml-code-nn':
                output = model(method_code=test_batch_data[0], beam_width=cf.beam_width, is_test=False,
                               uml_data=uml_batch, class_index=class_index, uml_index=uml_index)[-1]
            elif cf.modeltype == 'uml-h-deepcom':
                output = model(method_code=test_batch_data[0], method_sbt=test_batch_data[1],
                               beam_width=cf.beam_width, is_test=False, uml_data=uml_batch,
                               class_index=class_index, uml_index=uml_index)[-1]
            elif cf.modeltype == 'uml':
                    output = model(method_code=test_batch_data[0], method_sbt=test_batch_data[1],
                               method_summary=test_batch_data[2], use_teacher=False,
                               uml_data=uml_batch, class_index=class_index, uml_index=uml_index)[-1]

            # if cf.enable_sbt:
            #     output = model(method_code=test_batch_data[0], method_sbt=test_batch_data[1],
            #                    method_summary=test_batch_data[2], use_teacher=False,
            #                    uml_data = uml_batch, class_index = class_index, uml_index = uml_index)[-1]
            # else:
            #     output = model(method_code=test_batch_data[0], method_summary=test_batch_data[1], use_teacher=False,
            #                    uml_data = uml_batch, class_index = class_index, uml_index = uml_index)[-1]

            # output: batch_size, summary_length - 1, sum_vocab_size
            # pred: batch_size, summary_length - 1
            pred = torch.argmax(output, dim=2)

        # https://github.com/mcmillco/funcom/blob/41c737903a286e428493705e53c9c0e5946d2af6/bleu.py#L92
        # https://github.com/mcmillco/funcom/blob/41c737903a286e428493705e53c9c0e5946d2af6/bleu.py#L107
        # If p or t contain <s>, </s>, <UNK>, <NULL>,
        # they will be included as output.
        for i in range(cf.batch_size):
            p = pred[i].cpu().tolist()
            if trimTilEOS:
                if cf.EOS_token_id in p:  # truncate at first </s>
                    p = p[:p.index(cf.EOS_token_id) + 1]

            p = [summary_i2w[value] for value in p if
                    value not in [cf.PAD_token_id, cf.SOS_token_id, cf.EOS_token_id, cf.UNK_token_id]]

            preds.append(p)
            # t = trg[i].cpu().tolist()
            # t = [value for value in t if
            #      value not in [cf.PAD_token_id, cf.SOS_token_id, cf.EOS_token_id, cf.UNK_token_id]]
            # trgs.append([t])
    model.train()
    return preds


def bleu_so_far(refs, preds):
    # https://www.nltk.org/api/nltk.translate.html#nltk.translate.bleu_score.SmoothingFunction

    c_bleu1 = corpus_bleu(refs, preds, weights=(1, 0, 0, 0))
    c_bleu2 = corpus_bleu(refs, preds, weights=(0.5, 0.5, 0, 0))
    c_bleu3 = corpus_bleu(refs, preds, weights=(1/3, 1/3, 1/3, 0))
    c_bleu4 = corpus_bleu(refs, preds, weights=(0.25, 0.25, 0.25, 0.25))

    i_bleu2 = corpus_bleu(refs, preds, weights=(0, 1, 0, 0))
    i_bleu3 = corpus_bleu(refs, preds, weights=(0, 0, 1, 0))
    i_bleu4 = corpus_bleu(refs, preds, weights=(0, 0, 0, 1))

    c_bleu1 = round(c_bleu1 * 100, 2)
    c_bleu2 = round(c_bleu2 * 100, 2)
    c_bleu3 = round(c_bleu3 * 100, 2)
    c_bleu4 = round(c_bleu4 * 100, 2)
    i_bleu2 = round(i_bleu2 * 100, 2)
    i_bleu3 = round(i_bleu3 * 100, 2)
    i_bleu4 = round(i_bleu4 * 100, 2)

    ret = ''
    # https://machinelearningmastery.com/calculate-bleu-score-for-text-python/
    ret += ('Cumulative 4-gram BLEU (BLEU-4): %s\n' % c_bleu4)
    ret += 'Cumulative 1/2/3-gram BLEU: {}, {}, {}\n'.format(c_bleu1, c_bleu2, c_bleu3)
    ret += 'Individual 2/3/4-gram BLEU: {}, {}, {}\n'.format(i_bleu2, i_bleu3, i_bleu4)


    sf = SmoothingFunction()
    all_score = 0.0
    count = 0
    for r, p in zip(refs, preds):
        # nltk bug: https://github.com/nltk/nltk/issues/2204
        if len(p) == 1:
            continue
        # i.e. sentence_bleu
        score = nltk.translate.bleu(r, p, smoothing_function=sf.method4)
        all_score += score
        count += 1

    emse_bleu = round(all_score/count * 100, 2)
    ret += 'EMSE BLEU: {}\n'.format(emse_bleu)

    r_str_list = []
    p_str_list = []
    for r, p in zip(refs, preds):
        if len(r[0]) == 0 or len(p) == 0:
            continue

        r_str_list.append([" ".join([str(token_id) for token_id in r[0]])])
        p_str_list.append(" ".join([str(token_id) for token_id in p]))

    bleu_list = smooth_bleu(r_str_list, p_str_list)
    ret += ('CodeBert Smooth Bleu: %f\n' % bleu_list[0])
    bleu_score = _bleu(r_str_list, p_str_list)
    ret += ('Other Smooth Bleu: %f\n' % bleu_score)

    return ret, c_bleu4, bleu_list[0]


# def rouge(refs, preds):
#     refs_flatten = [r[0] for r in refs]
#     ret = ('Rouge for %s functions\n' % (len(preds)))
#     return ret + Rouge().get_scores(preds, refs_flatten, avg=True)


def calc_all_metrics(preds, refs):
    ret, _, codeBert_bleu = bleu_so_far(refs, preds)
    # preds_stem, refs_stem = stem_and_replace_predict_target(preds, refs)
    # metetor_rouge_cider(refs, preds)
    ret += "---use stemming and replacing 'get' with 'return'---- \n"
    # ret_stem, _, codeBert_bleu_stem = bleu_so_far(refs_stem, preds_stem)
    # metetor_rouge_cider(refs, preds)
    ret_stem,codeBert_bleu_stem = "", 0
    return ret+ret_stem, codeBert_bleu, codeBert_bleu_stem, preds


def test_bleu():
    refs = [[[0, 1, 2, 3, 4, 5, 6]]]
    preds = [[0, 1, 2, 4, 5, 0, 0]]
    print(bleu_so_far(refs, preds))

    # if preds is empty
    refs = [[[0, 1, 2, 3, 4, 5, 6]]]
    preds = [[]]
    print(bleu_so_far(refs, preds))


# https://github.com/wanyao1992/code_summarization_public/tree/master/evaluation
# https://github.com/salaniz/pycocoevalcap
def metetor_rouge_cider(refs, preds):
    refs_dict = {}
    preds_dict = {}
    for i in range(len(preds)):
        preds_dict[i] = [" ".join(preds[i])]
        refs_dict[i] = [" ".join(refs[i][0])]
    score_Meteor, scores_Meteor = Meteor().compute_score(refs_dict, preds_dict)
    print("Meteor: ", score_Meteor)

    score_Rouge, scores_Rouge = Rouge().compute_score(refs_dict, preds_dict)
    print("ROUGe: ", score_Rouge)

    score_Cider, scores_Cider = Cider().compute_score(refs_dict, preds_dict)
    print("Cider: ", score_Cider)


if __name__ == '__main__':

    preds = [["I am enshi"], ["she is Jhon"], ["he was boy"]]
    refs = [[["I am enshi"]], [["Her name is Jhon"]], [["he is Tom"]]]
    metetor_rouge_cider(refs, preds)
    # output
    # Meteor: 0.3489572501732486
    # ROUGe: 0.6301369863013698
    # Cider: 3.5254589515247576
