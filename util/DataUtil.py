import gzip
import itertools
import json
import math
import os
import pickle
import random
import re
import subprocess
import numpy as np
import torch
from spiral import ronin
from torch.utils.data import DataLoader
from util.Config import Config as cf
from util.Config import Config as config
from util.Dataset import CodeSummaryDataset
from util.LoggerUtil import info_logger
from multiprocessing import cpu_count, Pool
import nltk.stem
import copy
from itertools import chain


def set_seed(seed=0):
    max_seed = (1 << 32) - 1
    random.seed(seed)
    np.random.seed(random.randint(0, max_seed))
    torch.manual_seed(random.randint(0, max_seed))
    torch.cuda.manual_seed(random.randint(0, max_seed))
    torch.cuda.manual_seed_all(random.randint(0, max_seed))
    torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    torch.backends.cudnn.deterministic = True


def padding(line, max_len, padding_id):
    line_len = len(line)
    if line_len < max_len:
        line += [padding_id] * (max_len - line_len)
    return line


def read_pickle_data(data_path):
    with open(data_path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset


def save_pickle_data(path_dir, filename, data):
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    with open(os.path.join(path_dir, filename), 'wb') as f:
        pickle.dump(data, f)
    print("write file to " + os.path.join(path_dir, filename))


def open_json(path):
    with open(path, 'r') as f:
        files = f.readlines()
    return files


def filter_punctuation_nl(str_arr):
    """
    filter out punctuation
    :param str_arr: a string array, for example: ['an', 'apple', '.']
    :return: a string array without punctuation, for example: ['an', 'apple']
    """
    return [re.sub('[?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`]', '', s) for s in str_arr]


def spiral_split(str_arr):
    """
    split tokens in a string array with spiral ronin.
    :param str_arr: a string array, for example: ['readUTF8stream']
    :return: ['read', 'utf8', 'stream']
    """
    ronin_split = ronin.split
    return list(chain(*[[tok.lower() for tok in ronin_split(s)] for s in str_arr]))


def split_and_filter_identifier(str_arr):
    return spiral_split(filter_punctuation_nl(str_arr))


def filter_punctuation_pl(sequence):
    tokens = []
    for s in sequence:
        # https://www.jianshu.com/p/4f476942dca8
        # https://stackoverflow.com/questions/5843518/remove-all-special-characters-punctuation-and-spaces-from-string
        s = re.sub('\W+', '', s).replace("_", '')
        if s:
            tokens.append(s)
    return tokens


def lower_case_str_arr(str_arr):
    return [tok.lower() for tok in str_arr]


def lower_sequence(sequences):
    lower_seq = {}
    for idx, items in sequences.items():
        sub_sequence = [tok.lower() for tok in items]
        lower_seq[idx] = sub_sequence
    return lower_seq


def code_tokens_split_identifier(sequence):
    tokens = []
    for s in sequence:
        sub_sequence = [tok for tok in ronin.split(s) if tok]
        tokens.extend(sub_sequence)
    return tokens


def code_tokens_replace_str_num(str_arr):
    tokens = []
    for s in str_arr:
        if s[0] == '"' and s[-1] == '"':
            tokens.append("<STRING>")
        elif s.isdigit():
            tokens.append("<NUM>")
        else:
            tokens.append(s)
    return tokens


def split_and_filter_code_tokens(sequence, is_substring=False):
    tokens = []
    for s in sequence:
        if s[0] == '"' and s[-1] == '"':
            tokens.append("<STRING>")
        elif s[0] in config.STOP_WORDS and s[-1] in config.STOP_WORDS:
            continue
        elif s.isdigit():
            tokens.append("<NUM>")
        elif is_substring:
            tokens.append(s)
            continue
        else:
            sub_sequence = [tok.lower() for tok in ronin.split(s)]
            tokens.extend(split_and_filter_code_tokens(sub_sequence, is_substring=True))
    return tokens


def split_and_filter_identifier_lists(seqs):
    return [split_and_filter_identifier(seq) for seq in seqs]


def split_and_filter_identifier_lists_parallel(seqs):
    cores = cpu_count()
    pool = Pool(cores)
    seqs_split = array_split(seqs, cores)
    results = pool.map(split_and_filter_identifier_lists, seqs_split)
    pool.close()
    pool.join()
    return list_flatten(results)


def get_all_tokens(data):
    # data is a dict: {idx: string_seq, ....}
    all_tokens = []
    for seq in data.values():
        all_tokens.extend(seq)
    return all_tokens


def count_word(token_word_count, tokens):
    for token in tokens:
        token_word_count[token.lower()] = token_word_count.get(token.lower(), 0) + 1
    return token_word_count


def print_in_out_info():
    print("code_vocab_size: %d" % config.code_vocab_size)
    print("summary_vocab_size: %d" % config.summary_vocab_size)
    print("Read from:")
    print(config.train_data_path)
    print(config.val_data_path)
    print(config.test_data_path)
    print("Output to:", config.data_dir + '/standard_data_test.pkl')
    print("-------------Start--------------")


def print_dataset_info(dataset):
    print("Read train data: pair %d, code_vocab_size %d, summary_vocab_size %d " % (
        len(dataset['ctrain']), len(dataset['datstok']['i2w']), len(dataset['comstok']['i2w'])))
    print("Read valid data: pair %d" % (len(dataset['cval'])))
    print("Read test data: pair %d" % (len(dataset['ctest'])))


def build_vocab(word_count, start_id, vocab_size=-1):
    """
    # if vocab_size is negative, all tokens will be used
    """
    w2i, i2w = {}, {}
    word_count_ord = sorted(word_count.items(), key=lambda item: item[1], reverse=True)
    if vocab_size <= 0 or len(word_count_ord) < vocab_size:
        vocab_size = len(word_count_ord)
    for i in range(vocab_size):
        w2i[word_count_ord[i][0]] = i + start_id
        i2w[i + start_id] = word_count_ord[i][0]
    return w2i, i2w


def build_vocab_with_pad_unk(word_count, start_id, vocab_size=-1):
    w2i, i2w = build_vocab(word_count, start_id, vocab_size)

    w2i[config.PAD_token] = config.PAD_token_id
    i2w[config.PAD_token_id] = config.PAD_token

    unk_id = len(w2i)
    w2i[config.UNK_token] = unk_id
    i2w[unk_id] = config.UNK_token
    return w2i, i2w


def build_vocab_with_pad_unk_sos_eos(word_count, start_id, vocab_size=-1):
    w2i, i2w = build_vocab(word_count, start_id, vocab_size)
    w2i[config.SOS_token] = config.SOS_token_id
    i2w[config.SOS_token_id] = config.SOS_token
    w2i[config.EOS_token] = config.EOS_token_id
    i2w[config.EOS_token_id] = config.EOS_token
    w2i[config.PAD_token] = config.PAD_token_id
    i2w[config.PAD_token_id] = config.PAD_token

    unk_id = len(w2i)
    w2i[config.UNK_token] = unk_id
    i2w[unk_id] = config.UNK_token
    return w2i, i2w


# For code, do splitting identifier and padding
def transform_code(line, code_w2i, code_unk_id):
    code_ids = [code_w2i.get(token.lower(), code_unk_id) for
                token in split_and_filter_identifier(line["code_tokens"])[:config.code_len]]
    code_ids = padding(code_ids, config.code_len, config.PAD_token_id)
    return code_ids


# For sbt, only do padding
def transform_sbt(line, sbt_w2i, sbt_unk_id):
    sbt_ids = [sbt_w2i.get(token.lower(), sbt_unk_id) for
               token in line["sbt_tokens"][:config.sbt_len]]
    sbt_ids = padding(sbt_ids, config.sbt_len, config.PAD_token_id)
    return sbt_ids


# For summary, do splitting identifier, padding and add SOS/EOS
def transform_train_summary(line, summary_w2i, summary_unk_id, summary_word_count):
    summary_ids = [summary_w2i.get(token.lower(), summary_unk_id)
                   for token in split_and_filter_identifier(line["docstring_tokens"])[:config.summary_len - 1]]
    summary_ids.insert(0, config.SOS_token_id)

    summary_word_count[config.SOS_token] = summary_word_count[config.SOS_token] + 1

    if len(summary_ids) < config.summary_len:
        summary_ids.append(config.EOS_token_id)
        summary_word_count[config.EOS_token] = summary_word_count[config.EOS_token] + 1

    summary_ids = padding(summary_ids, config.summary_len, config.PAD_token_id)
    return summary_ids, summary_word_count


def transform_summary(line, summary_w2i, summary_unk_id):
    summary_ids = [summary_w2i.get(token.lower(), summary_unk_id)
                   for token in split_and_filter_identifier(line["docstring_tokens"])[:config.summary_len - 1]]

    summary_ids.insert(0, config.SOS_token_id)

    if len(summary_ids) < config.summary_len:
        summary_ids.append(config.EOS_token_id)

    summary_ids = padding(summary_ids, config.summary_len, config.PAD_token_id)
    return summary_ids


def read_data_and_count_word(data_dir):
    code_word_count = {}
    summary_word_count = {}
    if config.is_pkl_data_flag:
        lines = {}  # save all of the train data in lines
        # count word frequency
        for r, d, f in os.walk(data_dir):
            for file_name in f:
                if file_name.endswith("method_class_in_package.pkl"):
                    # with open(os.path.join(r, file_name), 'r') as f:
                    print(os.path.join(r, file_name))
                    methods = pickle.load(open(os.path.join(r, file_name), 'rb'))
                    for m_id, method in methods.items():
                        code_word_count = count_word(code_word_count,
                                                     split_and_filter_identifier(method["code_tokens"])[
                                                     :config.code_len])
                        # For summary, the first token is "<s>". To keep len as config.summary_len, we subtract 1.
                        summary_word_count = count_word(summary_word_count,
                                                        split_and_filter_identifier(method["docstring_tokens"])[
                                                        :config.summary_len - 1])

                        lines[m_id] = method
    else:
        lines = []
        # count word frequency
        for r, d, f in os.walk(data_dir):
            for file_name in f:
                if file_name.endswith(".jsonl.gz"):
                    with gzip.open(os.path.join(r, file_name), 'r') as file:
                        print(os.path.join(r, file_name))
                        for line in file:
                            line = json.loads(line)
                            code_word_count = count_word(code_word_count,
                                                         split_and_filter_identifier(line["code_tokens"])[
                                                         :config.code_len])
                            # For summary, the first token is "<s>". To keep len as config.summary_len, we subtract 1.
                            summary_word_count = count_word(summary_word_count,
                                                            split_and_filter_identifier(line["docstring_tokens"])[
                                                            :config.summary_len - 1])
                            lines.append(line)

    return lines, code_word_count, summary_word_count


def read_data_and_count_word_with_sbt(data_dir):
    code_word_count = {}
    summary_word_count = {}
    sbt_word_count = {}

    if config.is_pkl_data_flag:
        lines = {}  # save all of the train data in lines
        # count word frequency
        for r, d, f in os.walk(data_dir):
            for file_name in f:
                if file_name.endswith("method_class_in_package.pkl"):
                    # with open(os.path.join(r, file_name), 'r') as f:
                    print(os.path.join(r, file_name))
                    methods = pickle.load(open(os.path.join(r, file_name), 'rb'))
                    sbts = pickle.load(open(os.path.join(r, "sbts.pkl"), 'rb'))
                    for m_id, line in methods.items():
                        if m_id not in sbts.keys():
                            continue
                        code_word_count = count_word(code_word_count,
                                                     split_and_filter_identifier(line["code_tokens"])[:config.code_len])
                        # For summary, the first token is "<s>". To keep len as config.summary_len, we subtract 1.
                        summary_word_count = count_word(summary_word_count,
                                                        split_and_filter_identifier(line["docstring_tokens"])[
                                                        :config.summary_len - 1])
                        line["sbt_tokens"] = sbts[m_id].split(" ")
                        sbt_word_count = count_word(sbt_word_count, line["sbt_tokens"])
                        lines[m_id] = line
    else:
        lines = []
        # count word frequency
        for r, d, f in os.walk(data_dir):
            for file_name in f:
                if file_name.endswith(".jsonl.gz"):
                    with gzip.open(os.path.join(r, file_name), 'r') as file:
                        print(os.path.join(r, file_name))
                        for line in file:
                            line = json.loads(line)
                            code_word_count = count_word(code_word_count,
                                                         split_and_filter_identifier(line["code_tokens"])[
                                                         :config.code_len])
                            # For summary, the first token is "<s>". To keep len as config.summary_len, we subtract 1.
                            summary_word_count = count_word(summary_word_count,
                                                            split_and_filter_identifier(line["docstring_tokens"])[
                                                            :config.summary_len - 1])
                            lines.append(line)
                            # ToDo: add sbt

    return lines, code_word_count, summary_word_count, sbt_word_count


def read_data_append_id(data_dir):
    lines = []
    for r, d, f in os.walk(data_dir):
        for file_name in f:
            if file_name.endswith(".jsonl.gz"):
                with gzip.open(os.path.join(r, file_name), 'r') as file:
                    import chardet
                    print(os.path.join(r, file_name))
                    for line in file:
                        # t = chardet.detect(line).get("encoding")
                        line = json.loads(line)
                        line['id'] = len(lines)
                        lines.append(line)
    return lines


def extract_csn_data_append_id(data_dir):
    results = []
    for r, d, f in os.walk(data_dir):
        for file_name in f:
            if file_name.endswith(".jsonl.gz"):
                with gzip.open(os.path.join(r, file_name), 'r') as file:
                    print(os.path.join(r, file_name))
                    for line in file:
                        result = {}
                        line = json.loads(line)
                        result['id'] = len(results)
                        result['code'] = line['code']
                        result['code_tokens'] = line['code_tokens']
                        result['docstring'] = line['docstring']
                        result['docstring_tokens'] = line['docstring_tokens']
                        results.append(result)
    return results


def read_data_append_id_and_count_word(data_dir):
    lines = []
    code_word_count, summary_word_count = {}, {}
    for r, d, f in os.walk(data_dir):
        for file_name in f:
            if file_name.endswith(".jsonl.gz"):
                with gzip.open(os.path.join(r, file_name), 'r') as file:
                    print(os.path.join(r, file_name))
                    for line in file:
                        line = json.loads(line)
                        line['id'] = len(lines)
                        code_word_count = count_word(code_word_count,
                                                     split_and_filter_identifier(line["code_tokens"])[:config.code_len])
                        # For summary, the first token is "<s>". To keep len as config.summary_len, we subtract 1.
                        summary_word_count = count_word(summary_word_count,
                                                        split_and_filter_identifier(line["docstring_tokens"])[
                                                        :config.summary_len - 1])
                        lines.append(line)
    return lines, code_word_count, summary_word_count


def load_vocab(vocab_path, vocab_size={}):
    vocab = json.load(open(vocab_path, 'r'))
    summary_w2i, summary_i2w = filter_vocab_size(vocab['summary_w2i'], vocab_size.get("summary", -1))
    code_w2i, code_i2w = filter_vocab_size(vocab['code_w2i'], vocab_size.get("code", -1))
    sbt_w2i, sbt_i2w = filter_vocab_size(vocab['sbt_w2i'], vocab_size.get("sbt", -1))
    return vocab['code_word_count'], vocab['summary_word_count'], vocab['sbt_word_count'], \
           summary_w2i, summary_i2w, code_i2w, code_w2i, sbt_i2w, sbt_w2i


# build vocabulary only run first time
def build_and_save_vocab(data_path, vocab_path):
    # data_path = './csn_extracted_append_id/train/data.pkl'
    code_word_count, summary_word_count = {}, {}
    data = read_pickle_data(data_path)
    for line in data:
        code_word_count = count_word(code_word_count, split_and_filter_identifier(line["code_tokens"]))
        summary_word_count = count_word(summary_word_count, split_and_filter_identifier(line["docstring_tokens"]))

    # build code vocab, include <NULL> and <UNK>. Start after <NULL>=0.
    code_w2i, code_i2w = build_vocab_with_pad_unk(code_word_count, 1, config.code_vocab_size - 2)
    # build summary vocab, include <s>, </s>, <NULL> and <UNK>. Start after </s>=2.
    summary_w2i, summary_i2w = build_vocab_with_pad_unk_sos_eos(summary_word_count, 3, config.summary_vocab_size - 4)

    vocab_info = {'code_word_count': code_word_count,
                  'summary_word_count': summary_word_count,
                  'summary_w2i': summary_w2i,
                  'summary_i2w': summary_i2w,
                  'code_i2w': code_i2w,
                  'code_w2i': code_w2i}

    with open(vocab_path, 'w') as f:
        json.dump(vocab_info, f)


def label_encode_pkl_data(data_dir, code_w2i, summary_w2i, code_unk_id, summary_unk_id, sbt_w2i={}):
    method_code = {}
    method_summary = {}

    if config.using_sbt_flag:
        method_sbt = {}
        sbt_unk_id = sbt_w2i[config.UNK_token]

    for r, d, f in os.walk(data_dir):
        for file_name in f:
            if file_name.endswith("method_class_in_package.pkl"):
                print(os.path.join(r, file_name))
                lines = pickle.load(open(os.path.join(r, file_name), 'rb'))
                if config.using_sbt_flag:
                    sbts = pickle.load(open(os.path.join(r, "sbts.pkl"), 'rb'))

                for m_id, line in lines.items():
                    if config.using_sbt_flag:
                        if m_id not in sbts.keys():
                            continue
                        line["sbt_tokens"] = sbts[m_id].split(" ")
                        method_sbt[m_id] = transform_sbt(line, sbt_w2i, sbt_unk_id)

                    method_code[m_id] = transform_code(line, code_w2i, code_unk_id)
                    method_summary[m_id] = transform_summary(line, summary_w2i, summary_unk_id)

    if config.using_sbt_flag:
        return method_code, method_summary, method_sbt
    else:
        method_sbt = {}
        return method_code, method_summary, method_sbt


def label_encode_json_data(data_dir, code_w2i, summary_w2i, code_unk_id, summary_unk_id, sbt_w2i={}):
    method_code = {}
    method_summary = {}
    method_sbt = {}

    # if config.using_sbt_flag:
    #     method_sbt = {}
    #     sbt_unk_id = sbt_w2i[config.UNK_token]

    for r, d, f in os.walk(data_dir):
        for file_name in f:
            if file_name.endswith(".jsonl.gz"):
                with gzip.open(os.path.join(r, file_name), 'r') as file:
                    print(os.path.join(r, file_name))
                    for line in file:
                        line = json.loads(line)
                        method_code[len(method_code)] = transform_code(line, code_w2i, code_unk_id)
                        method_summary[len(method_summary)] = transform_summary(line, summary_w2i, summary_unk_id)
                        # Todo add sbt

    if config.using_sbt_flag:
        return method_code, method_summary, method_sbt
    else:
        method_sbt = {}
        return method_code, method_summary, method_sbt


def process_sbt_token(token, xml_tokens, index, node_type):
    continue_flag = False
    terminal_node_flag = False

    # Replacing '<type' with '(type' for non-terminal
    # Replacing '<\type' with 'type)' for non-terminal
    if node_type == "non_terminal":
        if token[:2] == '</':
            new_token = token.replace('</', ') ')
        else:
            new_token = token.replace('<', '( ')
    elif node_type == "terminal":
        new_token = ') ' + xml_tokens[index - 1].replace('<', '') + '_' + token
        terminal_node_flag = True
    else:
        literal_type = {
            'number': '<NUM>',
            'string': '<STR>',
            'null': 'null',
            'char': '<STR>',
            'boolean': token
        }
        new_token = ') ' + xml_tokens[index - 2].replace('<', '') + '_' + node_type + '_' + literal_type[
            node_type]
        terminal_node_flag = True

    return new_token, terminal_node_flag


def verify_node_type(token, xml_tokens, index, literal_list):
    """
    non_terminal:
        <type>
        </type>
    terminal:
        <type> token </type>  (this is right)
        <type> token <value> ... ( this is wrong)
    literal:
        <literal type="String" token </literal> | number | char | string | null | boolean
    """
    try:
        if token[0] == '<':
            node_type = 'non_terminal'
        elif xml_tokens[index - 1][1:] == xml_tokens[index + 1][2:] and xml_tokens[index - 1][0] == '<':
            node_type = 'terminal'
        elif xml_tokens[index - 1][:5] == 'type=':
            token_type = xml_tokens[index - 1].replace('type=', '')
            token_type = token_type.replace('\"', '')
            if token_type in literal_list:
                node_type = token_type
            else:
                node_type = None
        else:
            node_type = None
        return node_type
    except IndexError as e:
        print(e)
        return None


# Given the AST generating by http://131.123.42.38/lmcrs/beta/ ,
# it return SBT proposed by https://xin-xia.github.io/publication/icpc182.pdf
def xml2sbt(xml):
    # Replacing '<...>' with ' <...'
    xml = xml.replace('<', ' <')
    xml = xml.replace('>', ' ')

    #  splitting xml and filtering ''
    xml_tokens = xml.split(' ')
    xml_tokens = [i for i in xml_tokens if i != '']

    sbt = []
    terminal_node_flag = False
    literal_list = ['number', 'string', 'null', 'char', 'boolean']
    for i in range(len(xml_tokens)):

        # i = i+1 is unavailable in for loop, so we set terminal_node_flag to skip
        # terminal_nodes that have already been processed
        if terminal_node_flag:
            terminal_node_flag = False
            continue
        token = xml_tokens[i]
        node_type = verify_node_type(token, xml_tokens, i, literal_list)
        if node_type:
            new_token, terminal_node_flag = process_sbt_token(token, xml_tokens, i, node_type)
            sbt.append(new_token)
        else:
            continue
    return sbt


#  Make a new directory if it is not exist.
def make_directory(path):
    is_exist = os.path.exists(path)
    if not is_exist:
        os.makedirs(path)
    else:
        pass


def get_file_size(file_path):
    f_size = os.path.getsize(file_path)
    f_size = f_size / float(1024)
    return round(f_size, 2)


# Obtaining the sbt of the java file
def sbt_parser(file_path):
    if os.path.isfile(file_path):
        commandline = 'srcml ' + file_path
    else:
        commandline = 'srcml -l Java -t "{}"'.format(file_path)
    # https://docs.python.org/3/library/subprocess.html
    # Window
    # xml, _ = subprocess.Popen(commandline, stdout=subprocess.PIPE,
    # stderr=subprocess.PIPE, text=True).communicate(timeout=10)
    xml, _ = subprocess.Popen(commandline.split(" "), stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate(
        timeout=10)
    try:
        xml = re.findall(r"(<unit .*</unit>)", xml.decode('utf-8'), re.S)[0]
    except:
        print(xml)
        xml = " "
    sbt = xml2sbt(xml)
    return ' '.join(sbt)


def print_time(time_cost):
    m, s = divmod(time_cost, 60)
    h, m = divmod(m, 60)
    # print("time_cost: %d" % (time_cost))
    print("time_cost: %02d:%02d:%02d" % (h, m, s))


def time_format(time_cost):
    m, s = divmod(time_cost, 60)
    h, m = divmod(m, 60)
    # print("time_cost: %d" % (time_cost))
    return "%02d:%02d:%02d" % (h, m, s)


def extract_col(data, col):
    return [d[col] for d in data.values()]


def extract(data):
    return [d for d in data.values()]


# def extract_docstring_tokens(data):
#     cores = cpu_count()
#     ids = list(data.keys())
#     ids_split = array_split(ids, cores)
#     docstring_split = []
#
#     for split in ids_split:
#         docstring_split.append([data[i]['docstring_tokens'] for i in split])
#
#     return docstring_split


# li = [[1, 2, 3], [4, 5, 6], [7], [8, 9]] -> [1,2,3,4,5,6,7,8,9]
def list_flatten(li):
    flatten = itertools.chain.from_iterable
    return list(flatten(li))


def train_enshengid_2_fid(old_id):
    return old_id - 1


def val_enshengid_2_fid(old_id):
    train_len = 454451
    return old_id - 1 + train_len


def test_enshengid_2_fid(old_id):
    train_len = 454451
    val_len = 15328
    return old_id - 1 + train_len + val_len


def filter_vocab_size(full_w2i, vocab_size):
    w2i = {}
    i2w = {}
    if vocab_size <= 0:
        vocab_size = len(full_w2i)
    sort_w2i = sorted(full_w2i.items(), key=lambda item: item[1])[:vocab_size - 1]
    for word, idx in sort_w2i:
        w2i[word] = idx
        i2w[idx] = word
    unk_id = len(w2i)
    w2i[config.UNK_token] = unk_id
    i2w[unk_id] = config.UNK_token
    return w2i, i2w


def array_split(original_data, core_num):
    data = []
    total_size = len(original_data)
    per_core_size = math.ceil(total_size / core_num)
    for i in range(core_num):
        lower_bound = i * per_core_size
        upper_bound = min((i + 1) * per_core_size, total_size)
        data.append(original_data[lower_bound:upper_bound])
    return data


# Write to java file
def write_source_code_to_java_file(path, method_id, method):
    java_path = os.path.join(path, str(method_id) + ".java")
    with open(java_path, "w") as f:
        f.write(method)
    return java_path


def save_json_data(path_dir, filename, data):
    make_directory(path_dir)
    path = os.path.join(path_dir, filename)
    print("save json in " + path)
    with open(path, 'w') as output:
        json.dump(data, output)


def str_to_bool(str_data):
    return True if str_data.lower() == 'true' else False


def read_funcom_format_data(path):
    data = read_pickle_data(path)

    # load train, valid, test data and vocabulary
    train_summary = data["ctrain"]
    train_code = data["dtrain"]

    # valid info
    val_summary = data["cval"]
    val_code = data["dval"]
    val_ids = list(data["cval"].keys())

    # test info
    test_summary = data["ctest"]
    test_code = data["dtest"]
    test_ids = list(data['ctest'].keys())

    # vocabulary info
    summary_vocab = data["comstok"]
    code_vocab = data["datstok"]

    # i2w info
    summary_token_i2w = summary_vocab["i2w"]
    code_token_i2w = code_vocab["i2w"]

    summary_vocab_size = data["config"]["comvocabsize"]
    code_vocab_size = data["config"]["datvocabsize"]

    train_sbt = None
    val_sbt = None
    test_sbt = None
    try:
        train_sbt = data["strain"]
        val_sbt = data["sval"]
        test_sbt = data["stest"]

        # sbt_token_dict = data["smltok"]
        # sbt_token_i2w = sbt_token_dict["i2w"]

        sbt_vocab_size = data["config"]["smlvocabsize"]

    except:
        sbt_vocab_size = -1

    summary_len = data["config"]["comlen"]

    cf.UNK_token_id = summary_vocab_size - 1

    # obtain DataLoader for iteration
    train_dataset = CodeSummaryDataset(summary=train_summary, code=train_code,
                                       sbt=train_sbt)
    val_dataset = CodeSummaryDataset(summary=val_summary, code=val_code,
                                     sbt=val_sbt)
    test_dataset = CodeSummaryDataset(summary=test_summary, code=test_code,
                                      sbt=test_sbt)
    train_data_loader = DataLoader(train_dataset, shuffle=True, batch_size=cf.batch_size, pin_memory=True,
                                       num_workers=cf.num_subprocesses, drop_last=True)
    val_data_loader = DataLoader(val_dataset, shuffle=False, batch_size=cf.batch_size, pin_memory=True,
                                 num_workers=cf.num_subprocesses, drop_last=True)
    test_data_loader = DataLoader(test_dataset, shuffle=False, batch_size=cf.batch_size, pin_memory=True,
                                  num_workers=cf.num_subprocesses, drop_last=True)

    return train_dataset, val_dataset, test_dataset, train_data_loader, val_data_loader, test_data_loader, \
           code_vocab_size, sbt_vocab_size, summary_vocab_size, summary_vocab, summary_len, val_ids, test_ids


def get_file_name_old():
    data_len_str = "dlen" + str(config.code_len) + "_clen" + str(config.summary_len) + "_slen" + str(config.sbt_len)
    vocab_size_str = "_dvoc" + str(config.code_vocab_size) + "_cvoc" + str(config.summary_vocab_size) \
                     + "_svoc" + str(config.sbt_vocab_size)
    code_processing_str = "_djl" + str(config.code_tokens_using_javalang_results + 0) + \
                          "_dfp" + str(config.code_filter_punctuation + 0) + \
                          "_dsi" + str(config.code_split_identifier + 0) + \
                          "_dlc" + str(config.code_lower_case + 0) + \
                          "_dr" + str(config.code_replace_string_num + 0)
    summary_processing_str = "_cfp" + str(config.summary_filter_punctuation + 0) + \
                             "_csi" + str(config.summary_split_identifier + 0) + \
                             "_cfd" + str(config.summary_filter_bad_cases + 0)
    uml_processing_str = "_ufd" + str(config.uml_filter_data_bad_cases + 0)

    filename = data_len_str + vocab_size_str + code_processing_str + summary_processing_str + uml_processing_str \
               + '_dataset.pkl'
    return filename


def get_file_name():
    data_len_str = "dlen" + str(config.code_len) + "_clen" + str(config.summary_len) + "_slen" + str(config.sbt_len)
    vocab_size_str = "_dvoc" + str(config.code_vocab_size) + "_cvoc" + str(config.summary_vocab_size) \
                     + "_svoc" + str(config.sbt_vocab_size)

    filename = data_len_str + vocab_size_str + '_dataset.pkl'
    return filename


def stem_and_replace_predict_target(predict, target):
    predict_copy = copy.deepcopy(predict)
    target_copy = copy.deepcopy(target)
    empty_cnt = 0
    new_predict = []
    new_target = []
    stemming = nltk.stem.SnowballStemmer('english')

    for i in range(len(predict_copy)):
        pred = copy.deepcopy(predict_copy[i])
        if len(pred) < 1:
            empty_cnt += 1
            continue
        tar = copy.deepcopy(target_copy[i][0])
        try:
            pred[0] = stemming.stem(pred[0])
            tar[0] = stemming.stem(tar[0])
        except:
            continue
        if pred[0] == "get":
            pred[0] = "return"
        if tar[0] == "get":
            tar[0] = "return"
        new_predict.append(pred)
        new_target.append([tar])
    print("empty_cnt", empty_cnt)
    return new_predict, new_target
