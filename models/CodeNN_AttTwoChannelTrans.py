# Summarizing source code using a neural attention model, ACL'16
import heapq
from operator import attrgetter
from queue import Queue

import torch.nn as nn
import torch
import torch.nn.functional as F

from util.EvaluateUtil import BeamSearchNode
from util.GPUUtil import move_to_device
from util.Config import Config as cf
from models.GAT import GAT
from models.HGATCrossChannel import HGAT
from models.GCN import GCN
from models.GCN3 import GCN as GCN3
from models.FCN import FCN
from models.HGCN import HGCN
from torch_scatter import scatter_mean


class CodeNNModelUML(nn.Module):
    def __init__(self, code_vocab_size, summary_vocab_size, summary_len):
        super(CodeNNModelUML, self).__init__()
        self.hidden_size = cf.rnn_hidden_size
        self.summary_vocab_size = summary_vocab_size
        self.summary_len = summary_len

        self.summary_embedding = nn.Sequential(nn.Embedding(summary_vocab_size, self.hidden_size),
                                               nn.Dropout(p=0.5))
        self.code_embedding = nn.Embedding(code_vocab_size, self.hidden_size)
        self.summary_lstm_cell = nn.LSTMCell(input_size=self.hidden_size, hidden_size=self.hidden_size)
        self.t_linear = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.h_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.predict = nn.Sequential(nn.Tanh(),
                                     nn.Linear(self.hidden_size, summary_vocab_size),
                                     nn.Dropout(0.5))

        if cf.class_only:
            self.gnn = FCN(cf.gnn_num_feature, cf.gnn_num_hidden, dropout=cf.gnn_dropout)
        else:
            self.gnn = globals()[cf.gnn_type](cf.gnn_num_feature, cf.gnn_num_hidden, dropout=cf.gnn_dropout,
                                              aggregation=cf.aggregation)

        self.class_fcn = FCN(cf.gnn_num_feature, cf.gnn_num_hidden, dropout=cf.gnn_dropout)

        self.query_trans_name = nn.Linear(cf.rnn_hidden_size, cf.rnn_hidden_size)
        self.query_trans_uml = nn.Linear(cf.rnn_hidden_size, cf.rnn_hidden_size)
        self.value_trans_name = nn.Linear(cf.rnn_hidden_size, cf.rnn_hidden_size)
        self.value_trans_uml = nn.Linear(cf.rnn_hidden_size, cf.rnn_hidden_size)

    def beam_search_bfs(self, method_code, beam_width):
        batch_size = method_code.size(0)
        code_emb = self.code_embedding(method_code)

        summary_outputs = torch.zeros(batch_size, self.summary_len - 1, self.summary_vocab_size)
        summary_outputs = move_to_device(summary_outputs)

        # Summary starts with <s>
        start_input = torch.LongTensor([cf.SOS_token_id])
        # 1
        start_input = move_to_device(start_input)

        # sentence by sentence
        # BFS
        for seq_index in range(batch_size):

            # code_len x hidden_size
            single_code_emb = code_emb[seq_index]

            # 1 x summary_emb_size
            root_summary_emb = self.summary_embedding(start_input)

            # 1 x hidden_size
            root_hidden_state, root_cell_state = self.summary_lstm_cell(root_summary_emb)

            # [1, code_seq_len] = [1, hidden_size] x [hidden_size, code_seq_len]
            t_attn = torch.mm(root_hidden_state, single_code_emb.permute(1, 0))
            t_attn = F.softmax(t_attn, dim=1)

            # [1, hidden_size] = [1, code_seq_len] x [code_seq_len, hidden_size]
            t = torch.mm(t_attn, single_code_emb)

            # [summary_vocab_size]
            raw_probs = self.predict(self.t_linear(t) + self.h_linear(root_hidden_state)).view(-1)

            root = BeamSearchNode(hidden_state=(root_hidden_state, root_cell_state), previous_node=None,
                                  token_id=start_input,
                                  cumulative_prob=0, length=0, next_prob_distribution=raw_probs)

            q = Queue()
            q.put(root)

            end_nodes = []

            while not q.empty():
                candidates = []  # candidates in each layer
                heapq.heapify(candidates)

                for _ in range(q.qsize()):
                    prev_node = q.get()

                    # if token_id == cf.EOS_token_id or node.length == self.summary_len - 1:
                    # To be consistent with other methods, do not stop during generation, truncate during evaluation.
                    if prev_node.length == self.summary_len - 1:
                        end_nodes.append(prev_node)
                        continue

                    # use log to avoid multiplication of probs
                    top_probs, top_indices = F.log_softmax(prev_node.next_prob_distribution, 0).topk(beam_width)

                    for k in range(beam_width):
                        index = top_indices[k]
                        prob = top_probs[k].item()
                        cumulative_prob = prev_node.cumulative_prob + prob

                        # 1 x emb_size
                        summary_emb = self.summary_embedding(index).view(1, -1)

                        # 1 x hidden_size
                        hidden_state, cell_state = self.summary_lstm_cell(summary_emb, prev_node.hidden_state)

                        # [1, code_seq_len] = [1, hidden_size] x [hidden_size, code_seq_len]
                        t_attn = torch.mm(hidden_state, single_code_emb.permute(1, 0))
                        t_attn = F.softmax(t_attn, dim=1)

                        # [1, hidden_size] = [1, code_seq_len] x [code_seq_len, hidden_size]
                        t = torch.mm(t_attn, single_code_emb)

                        # [summary_vocab_size]
                        raw_probs = self.predict(self.t_linear(t) + self.h_linear(hidden_state)).view(-1)

                        child = BeamSearchNode(hidden_state=(hidden_state, cell_state), previous_node=prev_node,
                                               token_id=index, cumulative_prob=cumulative_prob,
                                               length=prev_node.length + 1, next_prob_distribution=raw_probs)

                        heapq.heappush(candidates, child)

                topk_result = heapq.nlargest(beam_width, candidates)
                for node in topk_result:
                    q.put(node)

            current_token = max(end_nodes, key=(attrgetter('cumulative_prob')))

            token_position_index = self.summary_len - 1

            while current_token.previous_node is not None and token_position_index > 0:  # exclude the root (<s>)
                # batch_size, self.summary_len - 1, self.summary_vocab_size
                summary_outputs[seq_index, token_position_index - 1, :] = current_token.previous_node.next_prob_distribution
                current_token = current_token.previous_node
                token_position_index -= 1

        return summary_outputs

    def beam_search_greedy(self, method_code, beam_width):
        batch_size = method_code.size(0)
        code_emb = self.code_embedding(method_code)

        summary_outputs = torch.zeros(batch_size, self.summary_len - 1, self.summary_vocab_size)
        summary_outputs = move_to_device(summary_outputs)

        # Summary starts with <s>
        start_input = torch.LongTensor([cf.SOS_token_id])
        # 1
        start_input = move_to_device(start_input)

        # sentence by sentence
        # greedy search
        for seq_index in range(batch_size):

            # code_len x hidden_size
            single_code_emb = code_emb[seq_index]

            # 1 x summary_emb_size
            root_summary_emb = self.summary_embedding(start_input)

            # 1 x hidden_size
            root_hidden_state, root_cell_state = self.summary_lstm_cell(root_summary_emb)

            # [1, code_seq_len] = [1, hidden_size] x [hidden_size, code_seq_len]
            t_attn = torch.mm(root_hidden_state, single_code_emb.permute(1, 0))
            t_attn = F.softmax(t_attn, dim=1)

            # [1, hidden_size] = [1, code_seq_len] x [code_seq_len, hidden_size]
            t = torch.mm(t_attn, single_code_emb)

            # [summary_vocab_size]
            raw_probs = self.predict(self.t_linear(t) + self.h_linear(root_hidden_state)).view(-1)

            root = BeamSearchNode(hidden_state=(root_hidden_state, root_cell_state), previous_node=None,
                                  token_id=start_input,
                                  cumulative_prob=0, length=0, next_prob_distribution=raw_probs)

            candidates = [root]  # global candidates list, max size is beam width

            for seq_pos_index in range(1, self.summary_len):
                next_candidates = []
                for prev_node in candidates:
                    # if token_id == cf.EOS_token_id or node.length == self.summary_len - 1:
                    # To be consistent with other methods, do not stop during generation, truncate during evaluation.
                    if prev_node.length == self.summary_len - 1:
                        continue

                    if seq_pos_index == 1:
                        topk = beam_width
                    else:
                        topk = 1

                    # use log to avoid multiplication of probs
                    top_probs, top_indices = F.log_softmax(prev_node.next_prob_distribution, 0).topk(topk)

                    for k in range(topk):
                        index = top_indices[k]
                        prob = top_probs[k].item()
                        cumulative_prob = prev_node.cumulative_prob + prob

                        # 1 x emb_size
                        summary_emb = self.summary_embedding(index).view(1, -1)

                        # 1 x hidden_size
                        hidden_state, cell_state = self.summary_lstm_cell(summary_emb, prev_node.hidden_state)

                        # [1, code_seq_len] = [1, hidden_size] x [hidden_size, code_seq_len]
                        t_attn = torch.mm(hidden_state, single_code_emb.permute(1, 0))
                        t_attn = F.softmax(t_attn, dim=1)

                        # [1, hidden_size] = [1, code_seq_len] x [code_seq_len, hidden_size]
                        t = torch.mm(t_attn, single_code_emb)

                        # [summary_vocab_size]
                        raw_probs = self.predict(self.t_linear(t) + self.h_linear(hidden_state)).view(-1)

                        child = BeamSearchNode(hidden_state=(hidden_state, cell_state), previous_node=prev_node,
                                               token_id=index, cumulative_prob=cumulative_prob,
                                               length=prev_node.length + 1, next_prob_distribution=raw_probs)
                        next_candidates.append(child)

                candidates = next_candidates

            assert(len(candidates) == beam_width)

            best_index = 0
            best_pro = candidates[0].cumulative_prob
            for i in range(1, len(candidates)):
                if best_pro < candidates[i].cumulative_prob:
                    best_index = i
                    best_pro = candidates[i].cumulative_prob

            current_token = candidates[best_index]
            token_position_index = self.summary_len - 1

            # trace back
            while current_token.previous_node is not None and token_position_index > 0:  # exclude the root (<s>)
                # batch_size, self.summary_len - 1, self.summary_vocab_size
                summary_outputs[seq_index, token_position_index - 1, :] = current_token.previous_node.next_prob_distribution
                current_token = current_token.previous_node
                token_position_index -= 1

            assert(token_position_index == 0)

        return summary_outputs

    def forward(self, method_code, beam_width, is_test, uml_data = None, class_index = None, uml_index = None):
        """

        :param method_code:     batch_size x code_len
        :param beam_width:
        :param is_test:
        :return:
        """

        # method_code = [batch size, seq len]
        # uml channel
        full_class_encoder_output = self.gnn(uml_data)
        class_encoder_output = full_class_encoder_output.index_select(0, class_index)
        # class_encoder_output [batch_size, gnn_num_hidden]

        class_feature = self.class_fcn(uml_data.x.index_select(0, class_index))
        # class_encoder_output [batch_size, gnn_num_hidden]

        if beam_width > 0 and is_test and cf.beam_search_method != "none":

            if cf.beam_search_method == "bfs":
                summary_outputs = self.beam_search_bfs(method_code, beam_width)
            elif cf.beam_search_method == "greedy":
                summary_outputs = self.beam_search_greedy(method_code, beam_width)
            else:
                raise Exception("Unrecognized beam_search_method: ", str(cf.beam_search_method))

        else:
            code_emb = self.code_embedding(method_code)  # [batchsize * code_len * hidden_size]
            batch_size = method_code.size(0)

            summary_outputs = torch.zeros(batch_size, self.summary_len - 1, self.summary_vocab_size)
            summary_outputs = move_to_device(summary_outputs)

            # Summary starts with <s>
            previous_summary_token = torch.LongTensor([cf.SOS_token_id] * batch_size)
            previous_summary_token = move_to_device(previous_summary_token)

            for i in range(1, self.summary_len):
                previous_summary_emb = self.summary_embedding(previous_summary_token)
                if i == 1:
                    # batch_size, hidden_size
                    hidden_state, cell_state = self.summary_lstm_cell(previous_summary_emb)
                else:
                    hidden_state, cell_state = self.summary_lstm_cell(previous_summary_emb, (hidden_state, cell_state))

                # [batch_size, 1, hidden_size] * [batch size, hidden_size, code_seq_len] = [batch_size, 1, code_len]
                t_attn = torch.bmm(hidden_state.unsqueeze(1), code_emb.permute(0, 2, 1))
                t_attn = F.softmax(t_attn, dim=2)
                # [batch_size, hidden_size]
                # t = torch.bmm(t_attn, code_emb).view(-1, self.hidden_size)
                t = torch.bmm(t_attn, code_emb)  # [batch_size, 1, hidden_size]

                # uml
                # [batch_size, 1, hidden_size]
                class_encoder_output = class_encoder_output.view(class_encoder_output.shape[0], 1, -1)
                class_feature = class_feature.view(class_feature.shape[0], 1, -1)

                # [batch_size, 2, hidden_size]
                class_query = torch.cat(
                    (self.query_trans_uml(class_encoder_output), self.query_trans_name(class_feature)), dim=1)
                class_value = torch.cat(
                    (self.value_trans_uml(class_encoder_output), self.value_trans_name(class_feature)), dim=1)
                # pdb.set_trace()
                # [batch_size, 1, hidden_size] * [batch size, hidden_size, 2] = [batch_size, 1, 2]
                class_context_attn = torch.bmm(hidden_state.unsqueeze(1), class_query.permute(0, 2, 1))
                class_context_attn = F.softmax(class_context_attn, dim=2)  # [batch_size, 1, 2]
                # [batch_size, 1, 2] * [batch size, 2, hidden_size] = [batch_size, 1, hidden_size]
                class_context = torch.bmm(class_context_attn, class_value)

                final_context = torch.cat((t, class_context), dim=2)  # [batch_size, hidden_size * 2]
                final_context = final_context.view(final_context.shape[0], -1)
                # [batch_size, vocab_size]
                # output = self.predict(final_context)

                # [batch_size, summary_vocab_size]
                # summary_output = self.predict(self.t_linear(t) + self.h_linear(hidden_state))
                summary_output = self.predict(self.t_linear(final_context) + self.h_linear(hidden_state))
                # decoder_output = [batch size, vocab_size]
                summary_outputs[:, i - 1, :] = summary_output
                previous_summary_token = summary_output.argmax(1)

        return [summary_outputs]