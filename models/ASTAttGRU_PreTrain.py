#!/usr/bin/env python
#!-*-coding:utf-8 -*-
import torch.nn as nn
import torch
from models.AttGRU import GRUEncoder
from util.GPUUtil import move_to_device
from util.Config import Config as cf
from models.ASTAttGRU import ASTAttnGRUDecoder


class AstAttGRU_PreTrain(nn.Module):
    def __init__(self, code_vocab_size, sbt_vocab_size, summary_vocab_size, encoder):
        super(AstAttGRU_PreTrain, self).__init__()
        self.code_encoder = encoder
        self.ast_encoder = GRUEncoder(vocab_size=sbt_vocab_size, emd_dim=cf.sbt_dim,
                                      hidden_size=cf.rnn_hidden_size,
                                      debug_msg="ast_encoder")
        self.decoder = ASTAttnGRUDecoder(vocab_size=summary_vocab_size, emd_dim=cf.summary_dim,
                                         hidden_size=cf.rnn_hidden_size)

    def forward(self, method_code, source_mask, method_summary, method_sbt,use_teacher):
        ast_encoder_output, ast_encoder_hidden = self.ast_encoder(method_sbt, None)

        output = self.code_encoder(method_code, attention_mask=source_mask)
        token_encoder_output = output[0].contiguous()  # batch_size, 100, 768

        summary_length = method_summary.size(1)
        # TODO: batch size % gpu != 0
        decoder_outputs = torch.zeros(int(cf.batch_size/4), summary_length - 1, self.decoder.output_size)
        decoder_outputs = move_to_device(decoder_outputs)
        # Summary starts with <s>
        decoder_inputs = method_summary[:, 0]

        decoder_hidden = None  # [batch_size, hidden_size]
        for i in range(1, summary_length):
            decoder_output, decoder_hidden = self.decoder(decoder_inputs,
                                                          ast_encoder_hidden,
                                                          None,
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
