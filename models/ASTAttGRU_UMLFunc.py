#!/usr/bin/env python
#!-*-coding:utf-8 -*-
import torch.nn as nn
import torch
import torch.nn.functional as F
from models.AttGRU import GRUEncoder
from util.GPUUtil import move_to_device
from util.ConfigUml import Config as cf
from util.LoggerUtil import debug_logger

# from models.GAT import GAT
from models.HGAT import HGAT as GAT
from models.GCN import GCN
from models.GCN3 import GCN as GCN3
from torch_scatter import scatter_mean

import pdb
import random

class ASTAttnGRUDecoder(nn.Module):
    def __init__(self, vocab_size, emd_dim, hidden_size, enable_sbt = True, enable_uml = False, class_output_dim = None):
        super(ASTAttnGRUDecoder, self).__init__()
        self.output_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, emd_dim)
        self.gru_cell = nn.GRUCell(input_size=emd_dim, hidden_size=hidden_size)
        
        if enable_uml and enable_sbt:
            self.predict = nn.Sequential(nn.Linear(hidden_size * 3 + class_output_dim, hidden_size),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(hidden_size, vocab_size))
        elif enable_sbt:
            self.predict = nn.Sequential(nn.Linear(hidden_size * 3, hidden_size),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(hidden_size, vocab_size))
        elif enable_uml:
            self.predict = nn.Sequential(nn.Linear(hidden_size * 2 + class_output_dim, hidden_size),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(hidden_size, vocab_size))
        else:
            self.predict = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(hidden_size, vocab_size))
        self.enable_sbt = enable_sbt
        self.enable_uml = enable_uml

    def forward(self, method_summary_token, sbt_encoder_hidden, code_encoder_hidden, sbt_encoder_output,
                code_encoder_output, class_encoder_output, prev_hidden_states):

        # method_summary_token is a batch of n-th nodes in the batch sequences
        # method_summary_token = [batch_size]
        # hidden = [1, batch_size, hidden_size]
        # encoder_outputs = [batch size, seq_len, hidden_size]

        summary_embs = self.embedding(method_summary_token)  # [batch_size, emb_dim]
        hidden_states = self.gru_cell(summary_embs, prev_hidden_states)  # [batch_size, hidden_size]

        debug_logger("ASTAttGRUDecoder (1): summary_embedding.shape %s, hidden_states.shape %s" % (str(summary_embs.shape), str(hidden_states.shape)))
        # hidden_states = [batch_size, 1, hidden_size]
        expand_hidden_states = hidden_states.unsqueeze(1)

        # [batch_size, 1, hidden_size] * [batch size, hidden_size, seq_len] = [batch_size, 1, seq_len]
        txt_attn = torch.bmm(expand_hidden_states, code_encoder_output.permute(0, 2, 1))
        txt_attn = F.softmax(txt_attn, dim=2)  # [batch_size, 1, seq_len]
        # [batch_size, 1, seq_len] * [batch size, seq_len, hidden_size] = [batch_size, 1, hidden_size]
        txt_context = torch.bmm(txt_attn, code_encoder_output)

        if self.enable_sbt:
            # [batch_size, 1, hidden_size] * [batch size, hidden_size, seq_len] = [batch_size, 1, seq_len]
            ast_attn = torch.bmm(expand_hidden_states, sbt_encoder_output.permute(0, 2, 1))
            # [batch_size, 1, seq_len]
            ast_attn = F.softmax(ast_attn, dim=2)
            # [batch_size, 1, seq_len] * [batch size, seq_len, hidden_size] = [batch_size, 1, hidden_size]
            ast_context = torch.bmm(ast_attn, sbt_encoder_output)
        
        if self.enable_uml:
            class_encoder_output = class_encoder_output.view(class_encoder_output.shape[0], 1, -1)
        #print(class_encoder_output.shape)
        #pdb.set_trace()
        if self.enable_sbt and self.enable_uml:
            # [batch_size, 1, hidden_size * 3 + gnn_output_dim]
            context = torch.cat((txt_context, expand_hidden_states, ast_context, class_encoder_output), dim=2)
        elif self.enable_sbt:
            context = torch.cat((txt_context, expand_hidden_states, ast_context), dim=2)
        elif self.enable_uml:
            context = torch.cat((txt_context, expand_hidden_states, class_encoder_output), dim=2)
        else:
            context = torch.cat((txt_context, expand_hidden_states), dim=2)
        # [batch_size, hidden_size * 3 + gnn_output_dim]
        context = context.view(context.shape[0], -1)

        # [batch_size, vocab_size]
        output = self.predict(context)

        return output, hidden_states


class AstAttGRUModelUML(nn.Module):

    def __init__(self, code_vocab_size, sbt_vocab_size, summary_vocab_size, enable_sbt = True, enable_uml = True):
        super(AstAttGRUModelUML, self).__init__()

        self.code_encoder = GRUEncoder(vocab_size=code_vocab_size, emd_dim=cf.code_dim,
                                       hidden_size=cf.rnn_hidden_size,
                                       debug_msg="code_encoder")
        if enable_sbt:
            self.ast_encoder = GRUEncoder(vocab_size=sbt_vocab_size, emd_dim=cf.sbt_dim,
                                          hidden_size=cf.rnn_hidden_size,
                                          debug_msg="ast_encoder")
        if enable_uml:
            if cf.gnn_concat and cf.gnn_concat_uml:
                self.decoder = ASTAttnGRUDecoder(vocab_size=summary_vocab_size, emd_dim=cf.summary_dim,
                                                 hidden_size=cf.rnn_hidden_size, enable_sbt = enable_sbt,
                                                 enable_uml = enable_uml, 
                                                 class_output_dim = cf.gnn_num_hidden[-1] * 2 + cf.gnn_num_feature + cf.num_func_feature)
                
            elif cf.gnn_concat:
                self.decoder = ASTAttnGRUDecoder(vocab_size=summary_vocab_size, emd_dim=cf.summary_dim,
                                                 hidden_size=cf.rnn_hidden_size, enable_sbt = enable_sbt,
                                                 enable_uml = enable_uml, 
                                                 class_output_dim = cf.gnn_num_hidden[-1] + cf.gnn_num_feature + cf.num_func_feature)
            elif cf.gnn_concat_uml:
                self.decoder = ASTAttnGRUDecoder(vocab_size=summary_vocab_size, emd_dim=cf.summary_dim,
                                                 hidden_size=cf.rnn_hidden_size, enable_sbt = enable_sbt,
                                                 enable_uml = enable_uml, 
                                                 class_output_dim = cf.gnn_num_hidden[-1] * 2+ cf.num_func_feature)
            else:
                self.decoder = ASTAttnGRUDecoder(vocab_size=summary_vocab_size, emd_dim=cf.summary_dim,
                                                 hidden_size=cf.rnn_hidden_size, enable_sbt = enable_sbt,
                                                 enable_uml = enable_uml, 
                                                 class_output_dim = cf.gnn_num_hidden[-1]+ cf.num_func_feature)
        else:
            self.decoder = ASTAttnGRUDecoder(vocab_size=summary_vocab_size, emd_dim=cf.summary_dim,
                                             hidden_size=cf.rnn_hidden_size, enable_sbt = enable_sbt,
                                             enable_uml = enable_uml)
        
        if enable_uml:
            if cf.gnn_type == "GCN":
                self.gnn = GCN(cf.gnn_num_feature, cf.gnn_num_hidden, dropout = cf.gnn_dropout)
            elif cf.gnn_type == "GAT":
                self.gnn = GAT(cf.gnn_num_feature, cf.gnn_num_hidden, dropout = cf.gnn_dropout)
        
        self.enable_uml = enable_uml
        self.enable_sbt = enable_sbt

    def forward(self, method_code, method_summary, use_teacher, method_sbt = None, uml_data = None, class_index = None, uml_index = None, func_name = None):
        # methods_code = [batch_size, code_len]
        # methods_sbt = [batch_size, sbt_len]
        # method_summary = [batch_size, summary_len]
        # uml_data = pyg.Data(x = [num_nodes, num_features], edge_list = [2, num_edges])
        # class_index = [batch_size]

        if self.enable_sbt:
            ast_encoder_output, ast_encoder_hidden = self.ast_encoder(method_sbt, None)
            # encoder_output [batch_size, seq_len, hidden_size]
            # encoder_hidden [num_layers * num_directions, batch_size, hidden_size]

            token_encoder_output, token_encoder_hidden = self.code_encoder(method_code, ast_encoder_hidden)
            # encoder_output [batch_size, seq_len, hidden_size]
            # encoder_hidden [num_layers * num_directions ,batch_size,hidden_size]
        else:
            ast_encoder_output, ast_encoder_hidden = None, None
            token_encoder_output, token_encoder_hidden = self.code_encoder(method_code, None)
        
        if self.enable_uml:
            full_class_encoder_output = self.gnn(uml_data)
            class_encoder_output = full_class_encoder_output.index_select(0, class_index)
            #pdb.set_trace()
            class_encoder_output = torch.cat([class_encoder_output, func_name.squeeze()], dim = 1)
            # pdb.set_trace()
            if cf.gnn_concat and cf.gnn_concat_uml:
                uml_pooling = scatter_mean(full_class_encoder_output, uml_data.batch, dim = 0)
                try:
                    uml_pooling = uml_pooling.index_select(0, uml_index)
                except Exception:
                    pdb.set_trace()
                class_encoder_output = torch.cat([class_encoder_output, uml_pooling, uml_data.x.index_select(0, class_index)], dim=1)
            elif cf.gnn_concat:
                class_encoder_output = torch.cat([class_encoder_output, uml_data.x.index_select(0, class_index)], dim=1)
            elif cf.gnn_concat_uml:
                uml_pooling = scatter_mean(class_encoder_output, uml_data.batch, dim = 0).index_select(0, uml_index)    
                class_encoder_output = torch.cat([class_encoder_output, uml_pooling], dim=1)
                
            # class_encoder_output [batch_size, gnn_num_hidden]
        else:
            class_encoder_output = None

        summary_length = method_summary.size(1)
        decoder_outputs = torch.zeros(cf.batch_size, summary_length - 1, self.decoder.output_size)
        decoder_outputs = move_to_device(decoder_outputs)

        # Summary starts with <s>
        decoder_inputs = method_summary[:, 0]

        decoder_hidden = token_encoder_hidden[0]  # [batch_size, hidden_size]
        for i in range(1, summary_length):
            decoder_output, decoder_hidden = self.decoder(decoder_inputs,
                                                          ast_encoder_hidden,
                                                          token_encoder_hidden,
                                                          ast_encoder_output,
                                                          token_encoder_output,
                                                          class_encoder_output,
                                                          decoder_hidden)

            # decoder_output = [batch_size, vocab_size]
            decoder_outputs[:, i - 1, :] = decoder_output

            if i + 1 != summary_length:
                if use_teacher:
                    prob = random.random()
                    if prob > cf.schedual_sampling_prob:
                        decoder_inputs = method_summary[:, i]
                    else:
                        decoder_inputs = decoder_output.argmax(1)
                else:
                    decoder_inputs = decoder_output.argmax(1)

        return [decoder_outputs]
