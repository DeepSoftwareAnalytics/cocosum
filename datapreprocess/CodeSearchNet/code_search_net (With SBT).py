import sys

sys.path.append("../../")
import os
import gzip
import json
from util.Config import Config as config
from util.DataUtil import padding, save_pickle_data, build_vocab
import pickle
import time
from spiral import ronin
import string
import subprocess

"""
Convert original CodeSearchNet data (with SBT) to the format we use 
"""
 
def get_sbt(code):
    code = code.replace('</unit>', '')
    code = code.replace('<', ' <')
    code = code.replace('>', ' ')
    codes = code.split(' ')
    codes = [i for i in codes if i != '']
    res = []
    flag = 0
    for i in range(len(codes)):
        if flag == 1:
            flag = 0
            continue
        word = codes[i]
        if word == '':
            continue
        if word[:2] == '</':
            word = word.replace('</', ') ')
            word = word.replace('>', '')

        elif word[:2] != '</' and word[0] == '<':
            word = word.replace('<', '( ')
            word = word.replace('>', '')
        else:
            try:
                if codes[i-1][1:] == codes[i+1][2:]:

                    word = ') ' + codes[i-1].replace('<', '') + '_' + word
                    flag = 1
                else:
                    continue
            except:
                continue
        res.append(word)
    return res


def process_sbt(code_token):
    code = ' '.join(code_token)
    code = code.replace('\"', '')
    code = code.replace('\\', '')
    res = subprocess.getoutput('srcml -l Java -t "{}"'.format(code))
    res = res.replace(
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<unit xmlns="http://www.srcML.org/srcML/src" revision="1.0.0" language="Java">',
        '')
    res = get_sbt(res)
    return res

# Identifier splitting (also known as identifier name tokenization) is the task of breaking apart program identifier
# strings such as getInt or readUTF8stream into component tokens: [get, int] and [read, utf8, stream].
def split_identifier(sequence):
    toks = []
    for s in sequence:
        if s in string.punctuation:
            toks.append(s)
        else:
            toks.extend([tok.lower() for tok in ronin.split(s)])
    return toks


def count_word(token_word_count, tokens):
    for token in tokens:
        token_word_count[token] = token_word_count.get(token, 0) + 1
    return token_word_count


def read_train_data(file_path):

    code_w2i_dict = {}
    code_i2w_dict = {}
    code_word_count = {}

    summary_w2i_dict = {}
    summary_i2w_dict = {}
    summary_word_count = {}

    sbt_w2i_dict = {}
    sbt_i2w_dict = {}
    sbt_word_count = {}

    lines = []
    error = 0
    # count word frequency
    for r, d, f in os.walk(file_path):
        for file_name in f:
            if file_name.endswith(".jsonl.gz"):
                with gzip.open(os.path.join(r, file_name), 'r') as f:
                    print(os.path.join(r, file_name))
                    for line in f:
                        line = json.loads(line)

                        code_word_count = count_word(code_word_count,
                                                     split_identifier(line["code_tokens"])[:config.code_len])
                        # For summary, the first token is "<s>". To keep len as config.summary_len, we subtract 1.
                        summary_word_count = count_word(summary_word_count,
                                                        split_identifier(line["docstring_tokens"])[:config.summary_len - 1])
                        # For sbt, process sbt tree and add to line
                        try:
                            sbt_toen = process_sbt(line["code_tokens"])[:config.sbt_len]
                            sbt_word_count = count_word(sbt_word_count, sbt_toen)
                            line["sbt_tokens"] = sbt_toen
                            lines.append(line)
                        except:
                            error += 1
                            print("error: {}, lines:{}".format(error, len(lines)))
                            print(line["code_tokens"])


    # build vocabulary
    code_w2i_dict, code_i2w_dict = build_vocab(code_word_count, code_w2i_dict,
                                               code_i2w_dict, 1,  # include <NULL> and <UNK>. Start after <NULL>=0.
                                               config.code_vocab_size - 2)

    code_w2i_dict[config.PAD_token] = config.PAD_token_id
    code_i2w_dict[config.PAD_token_id] = config.PAD_token

    code_unk_id = len(code_w2i_dict)
    code_w2i_dict[config.UNK_token] = code_unk_id
    code_i2w_dict[code_unk_id] = config.UNK_token

    sbt_w2i_dict, sbt_i2w_dict = build_vocab(sbt_word_count, sbt_w2i_dict,
                                             sbt_i2w_dict, 1,
                                             config.sbt_vocab_size - 2)

    sbt_w2i_dict[config.PAD_token] = config.PAD_token_id
    sbt_i2w_dict[config.PAD_token_id] = config.PAD_token

    sbt_unk_id = len(sbt_w2i_dict)
    sbt_w2i_dict[config.UNK_token] = sbt_unk_id
    sbt_i2w_dict[sbt_unk_id] = config.UNK_token

    summary_w2i_dict, summary_i2w_dict = build_vocab(summary_word_count, summary_w2i_dict,
                                                     summary_i2w_dict, 3,  # include <s>, </s>, <NULL> and <UNK>. Start after </s>=2.
                                                     config.summary_vocab_size - 4)

    summary_w2i_dict[config.PAD_token] = config.PAD_token_id
    summary_i2w_dict[config.PAD_token_id] = config.PAD_token

    summary_w2i_dict[config.SOS_token] = config.SOS_token_id
    summary_i2w_dict[config.SOS_token_id] = config.SOS_token

    summary_w2i_dict[config.EOS_token] = config.EOS_token_id
    summary_i2w_dict[config.EOS_token_id] = config.EOS_token

    summary_unk_id = len(summary_w2i_dict)
    summary_w2i_dict[config.UNK_token] = summary_unk_id
    summary_i2w_dict[summary_unk_id] = config.UNK_token

    summary_word_count[config.SOS_token] = 0
    summary_word_count[config.EOS_token] = 0

    # map each token to an int id
    # For summary, add SOS and EOS, and do padding to token_len
    method_code = {}
    method_summary = {}
    method_sbt = {}

    for line in lines:
        # For code, only do padding
        mapped_code_token_ids = [code_w2i_dict.get(token, code_unk_id) for
                                   token in split_identifier(line["code_tokens"])[:config.code_len]]
        mapped_code_token_ids = padding(mapped_code_token_ids, config.code_len, config.PAD_token_id)
        method_code[len(method_code)] = mapped_code_token_ids

        # For sbt, only do padding
        mapped_sbt_token_ids = [sbt_w2i_dict.get(token, sbt_unk_id) for
                                token in line["sbt_tokens"]]
        mapped_sbt_token_ids = padding(mapped_sbt_token_ids, config.sbt_len, config.PAD_token_id)
        method_sbt[len(method_sbt)] = mapped_sbt_token_ids

        # For summary, do padding and add SOS/EOS
        mapped_summary_token_ids = [summary_w2i_dict.get(token,
                                                     summary_unk_id)
                                    for token in split_identifier(line["docstring_tokens"])[
                                                 :config.summary_len - 1]]
        mapped_summary_token_ids.insert(0, config.SOS_token_id)
        summary_word_count[config.SOS_token] = summary_word_count[config.SOS_token] + 1

        if len(mapped_summary_token_ids) < config.summary_len:
            mapped_summary_token_ids.append(config.EOS_token_id)
            summary_word_count[config.EOS_token] = summary_word_count[config.EOS_token] + 1

        mapped_summary_token_ids = padding(mapped_summary_token_ids, config.summary_len, config.PAD_token_id)
        method_summary[len(method_summary)] = mapped_summary_token_ids

    # calculate the size of vocabulary
    code_token_vocab_size = len(code_i2w_dict)
    summary_token_vocab_size = len(summary_i2w_dict)
    sbt_token_vocab_size = len(sbt_i2w_dict)

    print("Read train data: method-summary pair %d,  code_token_vocab_size %d, summary_token_vocab_size %d, sbt_token_vocab_size %d " % (
        len(method_summary), code_token_vocab_size, summary_token_vocab_size, sbt_token_vocab_size))

    return method_code, code_w2i_dict, code_i2w_dict, code_word_count, \
           method_summary, summary_w2i_dict, summary_i2w_dict, summary_word_count, \
           method_sbt, sbt_w2i_dict, sbt_i2w_dict, sbt_word_count


def read_test_data(file_path, code_w2i_dict, summary_w2i_dict, sbt_w2i_dict):

    method_code = {}
    method_summary = {}
    method_sbt = {}

    code_unk_id = code_w2i_dict[config.UNK_token]
    summary_unk_id = summary_w2i_dict[config.UNK_token]
    sbt_unk_id = sbt_w2i_dict[config.UNK_token]
    for r, d, f in os.walk(file_path):
        for file_name in f:
            if file_name.endswith(".jsonl.gz"):
                with gzip.open(os.path.join(r, file_name), 'r') as f:
                    print(os.path.join(r, file_name))
                    for line in f:
                        line = json.loads(line)
                        # If the token is not in the token map built on train set, we will use UNK token to indicate it
                        # For code, only do padding
                        mapped_code_token_ids = [
                            code_w2i_dict.get(token, code_unk_id)
                            for token in split_identifier(line["code_tokens"])[:config.code_len]
                        ]

                        mapped_code_token_ids = padding(mapped_code_token_ids,
                                                        config.code_len,
                                                        config.PAD_token_id)
                        method_code[len(method_code)] = mapped_code_token_ids

                        # For sbt, only do padding
                        sbt_tokens = process_sbt(line["code_tokens"])[:config.sbt_len]
                        mapped_sbt_token_ids = [sbt_w2i_dict.get(token, sbt_unk_id) for
                                                token in sbt_tokens]
                        mapped_sbt_token_ids = padding(mapped_sbt_token_ids, config.sbt_len, config.PAD_token_id)
                        method_sbt[len(method_sbt)] = mapped_sbt_token_ids

                        # For summary, do padding and add SOS/EOS
                        mapped_summary_token_ids = [
                            summary_w2i_dict.get(token, summary_unk_id)
                            for token in split_identifier(line["docstring_tokens"])[:config.summary_len - 1]
                        ]

                        mapped_summary_token_ids.insert(0, config.SOS_token_id)

                        if len(mapped_summary_token_ids) < config.summary_len:
                            mapped_summary_token_ids.append(config.EOS_token_id)

                        mapped_summary_token_ids = padding(
                            mapped_summary_token_ids,
                            config.summary_len,
                            config.PAD_token_id)

                        method_summary[len(method_summary)] = mapped_summary_token_ids

    print("Read test/valid data: method-summary pair %d" % (len(method_summary)))

    return method_code, method_summary, method_sbt


if __name__ == "__main__":

    print("code_vocab_size: %d" % config.code_vocab_size)
    print("summary_vocab_size: %d" % config.summary_vocab_size)
    print("sbt_vocab_size: %d" % config.sbt_vocab_size)
    print("Read from:")
    print(config.train_data_path)
    print(config.val_data_path)
    print(config.test_data_path)
    print("Output to:", config.data_dir + '/standard_data.pkl')
    print("-------------Start--------------")

    start = time.perf_counter()

    train_method_code, train_code_w2i_dict, train_code_i2w_dict, \
    train_code_word_count, train_method_summary, train_summary_w2i_dict, \
    train_summary_i2w_dict, train_summary_word_count, train_method_sbt, \
    train_sbt_w2i_dict, train_sbt_i2w_dict, train_sbt_word_count = read_train_data(file_path=config.train_data_path)

    val_method_code, val_method_summary, val_method_sbt = read_test_data(
                                        file_path=config.val_data_path,
                                        code_w2i_dict=train_code_w2i_dict,
                                        summary_w2i_dict=train_summary_w2i_dict,
                                        sbt_w2i_dict=train_sbt_w2i_dict)

    test_method_code, test_method_summary, test_method_sbt = read_test_data(
                                        file_path=config.test_data_path,
                                        code_w2i_dict=train_code_w2i_dict,
                                        summary_w2i_dict=train_summary_w2i_dict,
                                        sbt_w2i_dict=train_sbt_w2i_dict)

    dataset = {}

    # dict = {key:ndarray}

    # summary
    dataset["ctrain"] = train_method_summary
    dataset["cval"] = val_method_summary
    dataset["ctest"] = test_method_summary

    dataset["dtrain"] = train_method_code
    dataset["dval"] = val_method_code
    dataset["dtest"] = test_method_code

    dataset["strain"] = train_method_sbt
    dataset["sval"] = val_method_sbt
    dataset["stest"] = test_method_sbt
    # i2w: id to token mapping. 1: <s>, 2: </s>, ..., max_id: oov_index. 0 is padding token but it is not contained in the dict

    # Method Summary Tokens
    # dataset["comstok"] = {"i2w": dict, "w2i": dict, "word_count": dict}
    dataset["comstok"] = {"i2w": train_summary_i2w_dict, "w2i": train_summary_w2i_dict, "word_count": train_summary_word_count}

    # Method Code Tokens
    # dataset["datstok"] = {"i2w": dict, "w2i": dict, "word_count": dict}
    dataset["datstok"] = {"i2w": train_code_i2w_dict, "w2i": train_code_w2i_dict, "word_count": train_code_word_count}

    # Method SBT Tokens
    # dataset["smltok"] = {"i2w": dict, "w2i": dict, "word_count": dict}
    dataset["smltok"] = {"i2w": train_sbt_i2w_dict, "w2i": train_sbt_w2i_dict, "word_count": train_sbt_word_count}

    # dataset["config"] = {"datvocabsize": int, "comvocabsize": int, "smlvocabsize": int, "datlen": int, "comlen": int,
    #                      "smllen": int}

    dataset["config"] = {"datvocabsize": len(train_code_i2w_dict), "comvocabsize": len(train_summary_i2w_dict), "smlvocabsize": len(train_sbt_i2w_dict),
                         "datlen": config.code_len, "comlen": config.summary_len, "smllen": config.sbt_len}

    save_pickle_data(config.data_dir, 'standard_data_sbt.pkl', dataset)
    end = time.perf_counter()
    print("time: ", end - start)
