#!/usr/bin/env python
#!-*-coding:utf-8 -*-
import torch.nn as nn
import torch
import torch.nn.functional as F
from models.AttGRU import GRUEncoder
from util.GPUUtil import move_to_device
from util.Config import Config as cf
from util.LoggerUtil import debug_logger


class ASTAttnGRUDecoder(nn.Module):
    def __init__(self, vocab_size, emd_dim, hidden_size):
        super(ASTAttnGRUDecoder, self).__init__()
        self.output_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, emd_dim)
        self.gru_cell = nn.GRUCell(input_size=emd_dim, hidden_size=hidden_size)
        self.predict = nn.Sequential(nn.Linear(hidden_size * 3, hidden_size),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(hidden_size, vocab_size))

    def forward(self, method_summary_token, sbt_encoder_hidden, code_encoder_hidden, sbt_encoder_output,
                code_encoder_output, prev_hidden_states):

        # method_summary_token is a batch of n-th nodes in the batch sequences
        # method_summary_token = [batch_size]
        # hidden = [1, batch_size, hidden_size]
        # encoder_outputs = [batch size, seq_len, hidden_size]

        summary_embs = self.embedding(method_summary_token)  # [batch_size, emb_dim]
        hidden_states = self.gru_cell(summary_embs, prev_hidden_states)  # [batch_size, hidden_size]

        # debug_logger("ASTAttGRUDecoder (1): summary_embedding.shape %s, hidden_states.shape %s" % (str(summary_embs.shape), str(hidden_states.shape)))
        # hidden_states = [batch_size, 1, hidden_size]
        expand_hidden_states = hidden_states.unsqueeze(1)

        # [batch_size, 1, hidden_size] * [batch size, hidden_size, seq_len] = [batch_size, 1, seq_len]
        txt_attn = torch.bmm(expand_hidden_states, code_encoder_output.permute(0, 2, 1))
        txt_attn = F.softmax(txt_attn, dim=2)  # [batch_size, 1, seq_len]
        # [batch_size, 1, seq_len] * [batch size, seq_len, hidden_size] = [batch_size, 1, hidden_size]
        txt_context = torch.bmm(txt_attn, code_encoder_output)

        # [batch_size, 1, hidden_size] * [batch size, hidden_size, seq_len] = [batch_size, 1, seq_len]
        ast_attn = torch.bmm(expand_hidden_states, sbt_encoder_output.permute(0, 2, 1))
        # [batch_size, 1, seq_len]
        ast_attn = F.softmax(ast_attn, dim=2)
        # [batch_size, 1, seq_len] * [batch size, seq_len, hidden_size] = [batch_size, 1, hidden_size]
        ast_context = torch.bmm(ast_attn, sbt_encoder_output)

        # [batch_size, 1, hidden_size * 3]
        context = torch.cat((txt_context, expand_hidden_states, ast_context), dim=2)
        # [batch_size, hidden_size * 3]
        context = context.view(context.shape[0], -1)

        # [batch_size, vocab_size]
        output = self.predict(context)

        return output, hidden_states


class AstAttGRUModel(nn.Module):

    def __init__(self, code_vocab_size, sbt_vocab_size, summary_vocab_size):
        super(AstAttGRUModel, self).__init__()

        self.code_encoder = GRUEncoder(vocab_size=code_vocab_size, emd_dim=cf.code_dim,
                                       hidden_size=cf.rnn_hidden_size,
                                       debug_msg="code_encoder")
        self.ast_encoder = GRUEncoder(vocab_size=sbt_vocab_size, emd_dim=cf.sbt_dim,
                                      hidden_size=cf.rnn_hidden_size,
                                      debug_msg="ast_encoder")
        self.decoder = ASTAttnGRUDecoder(vocab_size=summary_vocab_size, emd_dim=cf.summary_dim,
                                         hidden_size=cf.rnn_hidden_size)

    def forward(self, method_code, method_sbt, method_summary, use_teacher):
        # methods_code = [batch_size, code_len]
        # methods_sbt = [batch_size, sbt_len]
        # method_summary = [batch_size, summary_len]

        ast_encoder_output, ast_encoder_hidden = self.ast_encoder(method_sbt, None)
        # encoder_output [batch_size, seq_len, hidden_size]
        # encoder_hidden [num_layers * num_directions, batch_size, hidden_size]

        token_encoder_output, token_encoder_hidden = self.code_encoder(method_code, ast_encoder_hidden)
        # encoder_output [batch_size, seq_len, hidden_size]
        # encoder_hidden [num_layers * num_directions ,batch_size,hidden_size]

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
                                                          decoder_hidden)

            # decoder_output = [batch_size, vocab_size]
            decoder_outputs[:, i - 1, :] = decoder_output

            if i + 1 != summary_length:
                if use_teacher:
                    decoder_inputs = method_summary[:, i]
                else:
                    decoder_inputs = decoder_output.argmax(1).detach()

        return [decoder_outputs]
