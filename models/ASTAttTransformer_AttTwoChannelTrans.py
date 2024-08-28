#!/usr/bin/env python
# !-*-coding:utf-8 -*-
import torch.nn as nn
import torch
from models.AttGRU import GRUEncoder
from util.GPUUtil import move_to_device
from util.Config import Config as cf
from util.LoggerUtil import debug_logger

from models.GAT import GAT
from models.HGATCrossChannel import HGAT
from models.GCN import GCN
from models.GCN3 import GCN as GCN3
from models.FCN import FCN
from models.HGCN import HGCN
from torch_scatter import scatter_mean

import random
from models.ASTAttGRU_AttTwoChannel import ASTAttnGRUDecoder
from models.ASTAttTransformer import TransformerEncoder


class AstAttTransformerModelUML(nn.Module):

    def __init__(self, code_vocab_size, sbt_vocab_size, summary_vocab_size):
        super(AstAttTransformerModelUML, self).__init__()

        self.code_encoder = TransformerEncoder(vocab_size=code_vocab_size,
                                                   hidden_size=cf.rnn_hidden_size, debug_msg="code_encoder")

        self.ast_encoder = GRUEncoder(vocab_size=sbt_vocab_size, emd_dim=cf.sbt_dim,
                                          hidden_size=cf.rnn_hidden_size, debug_msg="ast_encoder")

        self.decoder = ASTAttnGRUDecoder(vocab_size=summary_vocab_size, emd_dim=cf.summary_dim,
                                         hidden_size=cf.rnn_hidden_size)

        if cf.class_only:
            self.gnn = FCN(cf.gnn_num_feature, cf.gnn_num_hidden, dropout=cf.gnn_dropout)
        else:
            self.gnn = globals()[cf.gnn_type](cf.gnn_num_feature, cf.gnn_num_hidden, dropout=cf.gnn_dropout,
                                              aggregation=cf.aggregation)

        self.class_fcn = FCN(cf.gnn_num_feature, cf.gnn_num_hidden, dropout=cf.gnn_dropout)

    def forward(self, method_code, method_sbt, method_summary, use_teacher, uml_data=None, class_index=None,
                uml_index=None):
        # methods_code = [batch_size, code_len]
        # methods_sbt = [batch_size, sbt_len]
        # method_summary = [batch_size, summary_len]
        # uml_data = pyg.Data(x = [num_nodes, num_features], edge_list = [2, num_edges])
        # class_index = [batch_size]

        ast_encoder_output, ast_encoder_hidden = self.ast_encoder(method_sbt, None)
        # encoder_output [batch_size, seq_len, hidden_size]
        # encoder_hidden [num_layers * num_directions, batch_size, hidden_size]


        token_encoder_output = self.code_encoder(method_code)
        token_encoder_hidden = None
        # encoder_output [batch_size, seq_len, hidden_size]
        # encoder_hidden [num_layers * num_directions ,batch_size,hidden_size]

        full_class_encoder_output = self.gnn(uml_data)
        class_encoder_output = full_class_encoder_output.index_select(0, class_index)
        # class_encoder_output [batch_size, gnn_num_hidden]

        class_feature = self.class_fcn(uml_data.x.index_select(0, class_index))
        # class_encoder_output [batch_size, gnn_num_hidden]

        summary_length = method_summary.size(1)
        decoder_outputs = torch.zeros(cf.batch_size, summary_length - 1, self.decoder.output_size)
        decoder_outputs = move_to_device(decoder_outputs)

        # Summary starts with <s>
        decoder_inputs = method_summary[:, 0]
        decoder_hidden = None

        for i in range(1, summary_length):
            decoder_output, decoder_hidden = self.decoder(decoder_inputs,
                                                          ast_encoder_hidden,
                                                          token_encoder_hidden,
                                                          ast_encoder_output,
                                                          token_encoder_output,
                                                          class_encoder_output,
                                                          class_feature,
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
