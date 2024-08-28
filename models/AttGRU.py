#!/usr/bin/env python
# !-*-coding:utf-8 -*-
import torch.nn as nn
import torch
import torch.nn.functional as F
from util.GPUUtil import move_to_device
from util.LoggerUtil import debug_logger
from util.Config import Config as cf


class GRUEncoder(nn.Module):

    def __init__(self, vocab_size, emd_dim, hidden_size, debug_msg):
        super(GRUEncoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emd_dim)
        self.gru = nn.GRU(input_size=emd_dim, hidden_size=hidden_size, batch_first=True)
        self.hidden_size = hidden_size
        self.debug_msg = debug_msg

    def forward(self, input, hidden):
        # debug_logger("GRUEncoder (1): input.shape %s" % (str(input.shape)))  # [batch_size, seq_len]
        embedded = self.embedding(input)  # [batch_size, seq_len, emd_dim]
        # debug_logger("GRUEncoder (2): embedded.shape %s" % (str(embedded.shape)))
        # if hidden is not None:
        #     debug_logger("GRUEncoder (3): hidden.shape %s" % (str(hidden.shape)))

        output, hidden = self.gru(embedded, hidden)
        # output [batch_size, seq_len, hidden_size]
        # hidden [num_layers * num_directions, batch_size, hidden_size]
        return output, hidden


class AttGRUDecoder(nn.Module):

    def __init__(self, hidden_size, vocab_size, emd_dim):
        super(AttGRUDecoder, self).__init__()
        self.output_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, emd_dim)
        self.gru_cell = nn.GRUCell(input_size=emd_dim, hidden_size=hidden_size)
        self.predict = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(hidden_size, vocab_size))

    def forward(self, method_summary_token, code_encoder_hidden, code_encoder_output, prev_hidden_states):
        # method_summary_token is a batch of n-th nodes in the batch sequences
        # method_summary_token = [batch_size]
        # hidden = [1, batch_size, hidden_size]
        # encoder_outputs = [batch size, seq_len, hidden_size]

        # summary_embs = [batch_size, emb_dim]
        summary_embs = self.embedding(method_summary_token)

        # hidden_states = [batch_size, hidden_size]
        hidden_states = self.gru_cell(summary_embs, prev_hidden_states)

        debug_logger("AttGRUDecoder (1): summary_embs.shape %s, hidden_states.shape %s" % (
            str(summary_embs.shape), str(hidden_states.shape)))
        # hidden_states = [batch_size, 1, hidden_size]
        expand_hidden_states = hidden_states.unsqueeze(1)

        # [batch_size, 1, hidden_size] * [batch size, hidden_size, seq_len] = [batch_size, 1, seq_len]
        txt_attn = torch.bmm(expand_hidden_states, code_encoder_output.permute(0, 2, 1))
        # [batch_size, 1, seq_len]
        txt_attn = F.softmax(txt_attn, dim=2)
        # [batch_size, 1, seq_len] * [batch size, seq_len, hidden_size] = [batch_size, 1, hidden_size]
        txt_context = torch.bmm(txt_attn, code_encoder_output)

        # [batch_size, 1, hidden_size * 2]
        context = torch.cat((txt_context, expand_hidden_states), dim=2)
        # [batch_size, hidden_size * 2]
        context = context.view(context.shape[0], -1)

        # [batch_size, vocab_size]
        output = self.predict(context)

        return output, hidden_states


class AttGRUModel(nn.Module):

    def __init__(self, token_vocab_size, summary_vocab_size):
        super(AttGRUModel, self).__init__()

        self.code_encoder = GRUEncoder(vocab_size=token_vocab_size, emd_dim=cf.code_dim,
                                       hidden_size=cf.rnn_hidden_size,
                                       debug_msg="Code_Encoder")
        self.decoder = AttGRUDecoder(vocab_size=summary_vocab_size, emd_dim=cf.summary_dim,
                                     hidden_size=cf.rnn_hidden_size)

    def forward(self, method_code, method_summary, use_teacher):
        # method_code = [batch size, seq len]
        # method_summary = [batch size, seq len]
        code_encoder_output, code_encoder_hidden = self.code_encoder(method_code, None)
        # code_encoder_output [batch_size, seq_len, hidden_size]
        # code_encoder_hidden [num_layers * num_directions, batchsize, hidden_size]

        summary_length = method_summary.size(1)
        decoder_outputs = torch.zeros(cf.batch_size, summary_length - 1, self.decoder.output_size)
        if cf.multi_gpu:
            decoder_outputs = decoder_outputs.to("cuda")
        else:
            decoder_outputs = move_to_device(decoder_outputs)

        # Summary starts with <s>
        decoder_inputs = method_summary[:, 0]

        decoder_hidden = code_encoder_hidden[0]
        for i in range(1, summary_length):
            decoder_output, decoder_hidden = self.decoder(decoder_inputs,
                                                          code_encoder_hidden,
                                                          code_encoder_output, decoder_hidden)

            # decoder_output = [batch size, vocab_size]
            decoder_outputs[:, i - 1, :] = decoder_output

            if i + 1 != summary_length:
                if use_teacher:
                    decoder_inputs = method_summary[:, i]
                else:
                    decoder_inputs = decoder_output.argmax(1).detach()

        return [decoder_outputs]
