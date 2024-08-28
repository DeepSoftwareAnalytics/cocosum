# Deep code comment generation with hybrid lexical and syntactical information, EMSE'19
from operator import attrgetter
from queue import Queue
import heapq
import torch.nn as nn
import torch
import torch.nn.functional as F
from models.AttGRU import GRUEncoder
from util.EvaluateUtil import BeamSearchNode
from util.GPUUtil import move_to_device
from util.Config import Config as cf


class HDeepComModel(nn.Module):

    def __init__(self, code_vocab_size, sbt_vocab_size, summary_vocab_size, summary_len):
        super(HDeepComModel, self).__init__()

        self.hidden_size = cf.rnn_hidden_size
        self.summary_vocab_size = summary_vocab_size
        self.summary_len = summary_len

        self.summary_embedding = nn.Embedding(summary_vocab_size, cf.summary_dim)
        self.summary_gru_cell = nn.GRUCell(input_size=cf.summary_dim, hidden_size=cf.rnn_hidden_size)

        self.code_encoder = GRUEncoder(vocab_size=code_vocab_size, emd_dim=cf.code_dim,
                                       hidden_size=cf.rnn_hidden_size, debug_msg="code_encoder")
        self.ast_encoder = GRUEncoder(vocab_size=sbt_vocab_size, emd_dim=cf.sbt_dim,
                                      hidden_size=cf.rnn_hidden_size, debug_msg="ast_encoder")

        self.predict = nn.Sequential(nn.Tanh(),
                                     nn.Linear(self.hidden_size, summary_vocab_size))


    def beam_search_bfs(self, code_encoder_outputs, ast_encoder_outputs, method_code, beam_width):

        batch_size = method_code.size(0)

        summary_outputs = torch.zeros(batch_size, self.summary_len - 1, self.summary_vocab_size)
        summary_outputs = move_to_device(summary_outputs)

        # Summary starts with <s>
        start_input = torch.LongTensor([cf.SOS_token_id])
        # 1
        start_input = move_to_device(start_input)

        # sentence by sentence
        # BFS
        for seq_index in range(batch_size):

            code_encoder_output = code_encoder_outputs[seq_index]
            ast_encoder_output = ast_encoder_outputs[seq_index]

            # 1 x summary_emb_size
            root_summary_emb = self.summary_embedding(start_input)

            # 1 x hidden_size
            root_hidden_state = self.summary_gru_cell(root_summary_emb, None)

            # [1, code_seq_len] = [1, hidden_size] x [hidden_size, code_seq_len]
            code_attn = torch.mm(root_hidden_state, code_encoder_output.permute(1, 0))
            code_attn = F.softmax(code_attn, dim=1)

            # [1 x hidden_size] = [1, code_seq_len] x [code_seq_len, hidden_size]
            code_context_vector = torch.mm(code_attn, code_encoder_output)

            # [1, ast_seq_len] = [1, hidden_size] x [hidden_size, ast_seq_len]
            ast_attn = torch.mm(root_hidden_state, ast_encoder_output.permute(1, 0))
            ast_attn = F.softmax(ast_attn, dim=1)

            # [1 x hidden_size] = [1, ast_seq_len] x [ast_seq_len, hidden_size]
            ast_context_vector = torch.mm(ast_attn, ast_encoder_output)

            context = code_context_vector + ast_context_vector

            # [summary_voc_size]
            raw_probs = self.predict(context).view(-1)

            root = BeamSearchNode(hidden_state=root_hidden_state, previous_node=None, token_id=start_input,
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
                        summary_hidden_state = self.summary_gru_cell(summary_emb, prev_node.hidden_state)

                        # [1, code_seq_len] = [1, hidden_size] x [hidden_size, code_seq_len]
                        code_attn = torch.mm(summary_hidden_state, code_encoder_output.permute(1, 0))
                        code_attn = F.softmax(code_attn, dim=1)

                        # [1 x hidden_size] = [1, code_seq_len] x [code_seq_len, hidden_size]
                        code_context_vector = torch.mm(code_attn, code_encoder_output)

                        # [1, ast_seq_len] = [1, hidden_size] x [hidden_size, ast_seq_len]
                        ast_attn = torch.mm(summary_hidden_state, ast_encoder_output.permute(1, 0))
                        ast_attn = F.softmax(ast_attn, dim=1)

                        # [1 x hidden_size] = [1, ast_seq_len] x [ast_seq_len, hidden_size]
                        ast_context_vector = torch.mm(ast_attn, ast_encoder_output)

                        context = code_context_vector + ast_context_vector

                        # [summary_voc_size]
                        raw_probs = self.predict(context).view(-1)

                        child = BeamSearchNode(hidden_state=summary_hidden_state, previous_node=prev_node,
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


    def beam_search_greedy(self, code_encoder_outputs, ast_encoder_outputs, method_code, beam_width):

        batch_size = method_code.size(0)

        summary_outputs = torch.zeros(batch_size, self.summary_len - 1, self.summary_vocab_size)
        summary_outputs = move_to_device(summary_outputs)

        # Summary starts with <s>
        start_input = torch.LongTensor([cf.SOS_token_id])
        # 1
        start_input = move_to_device(start_input)

        # sentence by sentence
        # greedy search
        for seq_index in range(batch_size):

            code_encoder_output = code_encoder_outputs[seq_index]
            ast_encoder_output = ast_encoder_outputs[seq_index]

            # 1 x summary_emb_size
            root_summary_emb = self.summary_embedding(start_input)

            # 1 x hidden_size
            root_hidden_state = self.summary_gru_cell(root_summary_emb, None)

            # [1, code_seq_len] = [1, hidden_size] x [hidden_size, code_seq_len]
            code_attn = torch.mm(root_hidden_state, code_encoder_output.permute(1, 0))
            code_attn = F.softmax(code_attn, dim=1)

            # [1 x hidden_size] = [1, code_seq_len] x [code_seq_len, hidden_size]
            code_context_vector = torch.mm(code_attn, code_encoder_output)

            # [1, ast_seq_len] = [1, hidden_size] x [hidden_size, ast_seq_len]
            ast_attn = torch.mm(root_hidden_state, ast_encoder_output.permute(1, 0))
            ast_attn = F.softmax(ast_attn, dim=1)

            # [1 x hidden_size] = [1, ast_seq_len] x [ast_seq_len, hidden_size]
            ast_context_vector = torch.mm(ast_attn, ast_encoder_output)

            context = code_context_vector + ast_context_vector

            # [summary_voc_size]
            raw_probs = self.predict(context).view(-1)

            root = BeamSearchNode(hidden_state=root_hidden_state, previous_node=None, token_id=start_input,
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
                        summary_emb = self.summary_embedding(index).view(1, cf.summary_dim)

                        # 1 x hidden_size
                        summary_hidden_state = self.summary_gru_cell(summary_emb, prev_node.hidden_state)

                        # [1, code_seq_len] = [1, hidden_size] x [hidden_size, code_seq_len]
                        code_attn = torch.mm(summary_hidden_state, code_encoder_output.permute(1, 0))
                        code_attn = F.softmax(code_attn, dim=1)

                        # [1 x hidden_size] = [1, code_seq_len] x [code_seq_len, hidden_size]
                        code_context_vector = torch.mm(code_attn, code_encoder_output)

                        # [1, ast_seq_len] = [1, hidden_size] x [hidden_size, ast_seq_len]
                        ast_attn = torch.mm(summary_hidden_state, ast_encoder_output.permute(1, 0))
                        ast_attn = F.softmax(ast_attn, dim=1)

                        # [1 x hidden_size] = [1, ast_seq_len] x [ast_seq_len, hidden_size]
                        ast_context_vector = torch.mm(ast_attn, ast_encoder_output)

                        context = code_context_vector + ast_context_vector

                        # [summary_voc_size]
                        raw_probs = self.predict(context).view(-1)

                        child = BeamSearchNode(hidden_state=summary_hidden_state, previous_node=prev_node,
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


    def forward(self, method_code, method_sbt, beam_width, is_test):
        """

        :param method_code:     batch_size x code_len
        :param method_sbt:      batch_size x sbt_len
        :param beam_width:
        :param is_test:
        :return:
        """
        ast_encoder_outputs, ast_encoder_last_hidden = self.ast_encoder(method_sbt, None)
        # ast_encoder_outputs [batch_size, seq_len, hidden_size]
        # ast_encoder_last_hidden [num_layers * num_directions, batch_size, hidden_size]

        code_encoder_outputs, code_encoder_last_hidden = self.code_encoder(method_code, None)
        # code_encoder_outputs [batch_size, seq_len, hidden_size]
        # code_encoder_last_hidden [num_layers * num_directions ,batch_size,hidden_size]

        if beam_width > 0 and is_test and cf.beam_search_method != "none":

            if cf.beam_search_method == "bfs":
                summary_outputs = self.beam_search_bfs(code_encoder_outputs, ast_encoder_outputs, method_code, beam_width)
            elif cf.beam_search_method == "greedy":
                summary_outputs = self.beam_search_greedy(code_encoder_outputs, ast_encoder_outputs, method_code, beam_width)
            else:
                raise Exception("Unrecognized beam_search_method: ", str(cf.beam_search_method))

        else:

            batch_size = method_code.size(0)
            summary_outputs = torch.zeros(batch_size, self.summary_len - 1, self.summary_vocab_size)
            summary_outputs = move_to_device(summary_outputs)

            # Summary starts with <s>
            decoder_input = torch.LongTensor([cf.SOS_token_id] * batch_size)
            # batch_size
            decoder_input = move_to_device(decoder_input)

            pre_summary_hidden_state = None

            for token_index in range(1, self.summary_len):

                summary_embs = self.summary_embedding(decoder_input)

                # batch_size x hidden_size
                pre_summary_hidden_state = self.summary_gru_cell(summary_embs, pre_summary_hidden_state)

                # The paper does not explain what is the alignment model. Use product as codenn and AstAttGRU
                # [batch_size, 1, hidden_size] * [batch size, hidden_size, code_seq_len] = [batch_size, 1, code_seq_len]
                code_attn = torch.bmm(pre_summary_hidden_state.unsqueeze(1), code_encoder_outputs.permute(0, 2, 1))
                code_attn = F.softmax(code_attn, dim=2)
                # batch_size x hidden_size
                code_context_vector = torch.bmm(code_attn, code_encoder_outputs).view(-1, self.hidden_size)

                # [batch_size, 1, hidden_size] * [batch size, hidden_size, sbt_seq_len] = [batch_size, 1, sbt_seq_len]
                ast_attn = torch.bmm(pre_summary_hidden_state.unsqueeze(1), ast_encoder_outputs.permute(0, 2, 1))
                ast_attn = F.softmax(ast_attn, dim=2)
                # batch_size x hidden_size
                ast_context_vector = torch.bmm(ast_attn, ast_encoder_outputs).view(-1, self.hidden_size)

                context = code_context_vector + ast_context_vector

                # batch_size x summary_voc_size
                decoder_output = self.predict(context)

                summary_outputs[:, token_index - 1, :] = decoder_output

                if token_index + 1 != self.summary_len:
                    decoder_input = decoder_output.argmax(1)

        return [summary_outputs]
