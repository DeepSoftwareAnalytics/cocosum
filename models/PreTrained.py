"""
Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
"""
"""
Codes in Pretrained class and convert_examples_to_features functions are cited from MSRA CodeBert Team 
"""

from util.Config import Config as cf
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)
MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}


class Pretrained():
    def __init__(self):
        self.pretrained_modeltype = cf.pretrained_modeltype
        self.pretrained_modelpath = cf.pretrained_modelpath
        print(self.pretrained_modeltype)
        self.config_class, self.model_class, self.tokenizer_class = MODEL_CLASSES[self.pretrained_modeltype]
        self.config = self.config_class.from_pretrained(self.pretrained_modelpath)

    def get_tokenizer(self):
        tokenizer = self.tokenizer_class.from_pretrained(cf.tokenizer_name, do_lower_case=cf.code_lower_case)
        return tokenizer

    def get_encoder(self):
        encoder = self.model_class.from_pretrained(self.pretrained_modelpath, config=self.config)
        return encoder


def convert_examples_to_features(code_tokens, tokenizer):
    pretrain_ids = dict()
    pretrain_mask = dict()
    keys = list(code_tokens.keys())
    for key in keys:
        source_tokens = tokenizer.tokenize(code_tokens[key])[:cf.code_len-2]
        source_tokens = [tokenizer.cls_token] + source_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        source_mask = [1] * (len(source_tokens))

        padding_length = cf.code_len - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length
        source_mask += [0] * padding_length
        pretrain_ids[key] = source_ids
        pretrain_mask[key] = source_mask
    return pretrain_ids, pretrain_mask