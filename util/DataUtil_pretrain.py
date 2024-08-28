import torch
from util.Config import Config as cf
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from util.DataUtil import read_pickle_data
from models.PreTrained import Pretrained, convert_examples_to_features
from torch.utils.data.distributed import DistributedSampler

class CodeSummaryPreTrainDataset(Dataset):
    def __init__(self, summary, code, sbt=None, mask=None):
        super(CodeSummaryPreTrainDataset, self).__init__()
        self.seq_num = len(summary)

        assert (len(code) == self.seq_num)

        self.summary = []
        self.code = []
        self.mask = []

        if sbt is not None:
            assert (len(sbt) == self.seq_num)
            self.sbt = []
        else:
            self.sbt = None

        for key, seq in summary.items():
            self.summary.append(seq)
            self.code.append(code[key])
            self.mask.append(mask[key])

            if sbt is not None:
                self.sbt.append(sbt[key])
        self.summary = torch.LongTensor(self.summary)
        self.code = torch.LongTensor(self.code)
        self.mask = torch.LongTensor(self.mask)

        if self.sbt is not None:
            self.sbt = torch.LongTensor(self.sbt)

    def __len__(self):
        return self.seq_num

    def __getitem__(self, idx):
        if self.sbt is not None:
            return self.code[idx], self.mask[idx], self.summary[idx], self.sbt[idx]
        else:
            return self.code[idx], self.mask[idx], self.summary[idx]


class CodeSummaryUmlPreTrainDataset(Dataset):
    def __init__(self, summary, code, mask, sbt=None, mapping=None, func_name=None):
        super(CodeSummaryUmlPreTrainDataset, self).__init__()

        self.seq_num = len(summary)

        assert (len(code) == self.seq_num)

        self.summary = []
        self.code = []
        self.mask = []

        if sbt is not None:
            assert (len(sbt) == self.seq_num)
            self.sbt = []
        else:
            self.sbt = None

        if mapping is not None:
            # assert (len(mapping["method2uml"]) == self.seq_num)
            # assert (len(mapping["method2class"]) == self.seq_num)
            self.method2uml = []
            self.method2class = []
        else:
            self.method2uml = None
            self.method2class = None

        if func_name is not None:
            assert (len(func_name) == self.seq_num)
            assert (len(func_name) == self.seq_num)
            self.func_name = []
        else:
            self.func_name = None

        for key, seq in summary.items():
            self.summary.append(seq)
            self.code.append(code[key])
            self.mask.append(mask[key])
            if sbt is not None:
                self.sbt.append(sbt[key])
            if mapping is not None:
                self.method2uml.append(mapping["method2uml"][key])
                self.method2class.append(mapping["method2class"][key])
            if func_name is not None:
                self.func_name.append(func_name[key])

        self.summary = torch.LongTensor(self.summary)
        self.code = torch.LongTensor(self.code)
        self.mask = torch.LongTensor(self.mask)

        if self.sbt is not None:
            self.sbt = torch.LongTensor(self.sbt)
        if mapping is not None:
            self.method2uml = torch.LongTensor(self.method2uml)
            self.method2class = torch.LongTensor(self.method2class)
        if func_name is not None:
            self.func_name = torch.FloatTensor(self.func_name)

    def __len__(self):
        return self.seq_num

    def __getitem__(self, idx):
        if self.sbt is not None and self.method2uml is not None and self.func_name is not None:
            return self.code[idx], self.mask[idx], self.sbt[idx], self.func_name[idx], self.summary[idx], self.method2uml[idx], \
                   self.method2class[idx]
        elif self.sbt is not None and self.method2uml is not None:
            return self.code[idx], self.mask[idx], self.sbt[idx], self.summary[idx], self.method2uml[idx], self.method2class[idx]
        elif self.sbt is not None and self.func_name is not None:
            return self.code[idx], self.mask[idx], self.sbt[idx], self.func_name[idx], self.summary[idx]
        elif self.method2uml is not None and self.func_name is not None:
            return self.code[idx], self.mask[idx], self.func_name[idx], self.summary[idx], self.method2uml[idx], self.method2class[idx]
        elif self.sbt is not None:
            return self.code[idx], self.mask[idx], self.sbt[idx], self.summary[idx]
        elif self.method2uml is not None:
            return self.code[idx], self.mask[idx], self.summary[idx], self.method2uml[idx], self.method2class[idx]
        elif self.func_name is not None:
            return self.code[idx], self.mask[idx], self.func_name[idx], self.summary[idx]
        else:
            return self.code[idx], self.mask[idx], self.summary[idx]


def get_source_code(code_fids, source):
    source_code = dict()
    for i in code_fids:
        # The following two lines are cited from CodeBert team
        code = ' '.join(source[i]['code_tokens']).replace('\n', ' ')
        code = ' '.join(code.strip().split())
        source_code[i] = code
    return source_code


def read_pretrain_format_data(path, tokenizer):
    source_code = read_pickle_data(cf.source_code_path)
    data = read_pickle_data(path)

    # load train, valid, test data and vocabulary
    train_summary = data["ctrain"]
    train_code_ids = list(data["dtrain"].keys())

    # valid info
    val_summary = data["cval"]
    val_code_ids = list(data["dval"].keys())
    val_ids = list(data["cval"].keys())

    # test info
    test_summary = data["ctest"]
    test_code_ids = list(data["dtest"].keys())
    test_ids = list(data['ctest'].keys())

    # vocabulary info
    summary_vocab = data["comstok"]
    # code_vocab = data["datstok"]

    # i2w info
    # summary_token_i2w = summary_vocab["i2w"]
    # code_token_i2w = code_vocab["i2w"]

    # get source code from code_ids
    train_source_code = get_source_code(train_code_ids, source_code['train'])
    test_source_code = get_source_code(test_code_ids, source_code['test'])
    val_source_code = get_source_code(val_code_ids, source_code['val'])

    # get ids from pre-trained model vocab
    train_code, train_mask = convert_examples_to_features(train_source_code, tokenizer)
    test_code, test_mask = convert_examples_to_features(test_source_code, tokenizer)
    val_code, val_mask = convert_examples_to_features(val_source_code, tokenizer)

    summary_vocab_size = data["config"]["comvocabsize"]
    code_vocab_size = data["config"]["datvocabsize"]

    train_sbt = None
    val_sbt = None
    test_sbt = None
    sbt_vocab_size = -1
    if cf.modeltype == 'ast-att-gru-codebert' or cf.modeltype == 'h-deepcom':
        train_sbt = data["strain"]
        val_sbt = data["sval"]
        test_sbt = data["stest"]

        sbt_vocab_size = data["config"]["smlvocabsize"]

    summary_len = data["config"]["comlen"]

    cf.UNK_token_id = summary_vocab_size - 1

    # obtain DataLoader for iteration
    train_dataset = CodeSummaryPreTrainDataset(summary=train_summary, code=train_code,
                                       sbt=train_sbt, mask=train_mask)
    val_dataset = CodeSummaryPreTrainDataset(summary=val_summary, code=val_code,
                                     sbt=val_sbt, mask=val_mask)
    test_dataset = CodeSummaryPreTrainDataset(summary=test_summary, code=test_code,
                                      sbt=test_sbt, mask=test_mask)

    train_data_loader = DataLoader(train_dataset, shuffle=True, batch_size=cf.batch_size, pin_memory=True,
                                   num_workers=cf.num_subprocesses, drop_last=True)
    val_data_loader = DataLoader(val_dataset, shuffle=False, batch_size=cf.batch_size, pin_memory=True,
                                 num_workers=cf.num_subprocesses, drop_last=True)
    test_data_loader = DataLoader(test_dataset, shuffle=False, batch_size=cf.batch_size, pin_memory=True,
                                  num_workers=cf.num_subprocesses, drop_last=True)

    return train_dataset, val_dataset, test_dataset, train_data_loader, val_data_loader, test_data_loader, \
           code_vocab_size, sbt_vocab_size, summary_vocab_size, summary_vocab, summary_len, val_ids, test_ids


def read_uml_pretrain_format_data(path, uml_path, tokenizer):
    data = read_pickle_data(path)
    source_code = read_pickle_data(cf.source_code_path)
    # load uml data
    uml = torch.load(uml_path)
    uml_train = uml["train"]
    uml_val = uml["val"]
    uml_test = uml["test"]

    train_m2u = data["m2utrain"]
    val_m2u = data["m2uval"]
    test_m2u = data["m2utest"]

    train_m2c = data["m2ctrain"]
    val_m2c = data["m2cval"]
    test_m2c = data["m2ctest"]

    # load train, valid, test data and vocabulary
    train_summary = data["ctrain"]
    train_code_ids = list(data["dtrain"].keys())

    # valid info
    val_summary = data["cval"]
    val_code_ids = list(data["dval"].keys())
    val_ids = list(data["cval"].keys())

    # test info
    test_summary = data["ctest"]
    test_code_ids = list(data["dtest"].keys())
    test_ids = list(data['ctest'].keys())

    # vocabulary info
    summary_vocab = data["comstok"]
    # code_vocab = data["datstok"]

    # i2w info
    # summary_token_i2w = summary_vocab["i2w"]
    # code_token_i2w = code_vocab["i2w"]

    # get source code from code_ids
    train_source_code = get_source_code(train_code_ids, source_code['train'])
    test_source_code = get_source_code(test_code_ids, source_code['test'])
    val_source_code = get_source_code(val_code_ids, source_code['val'])

    # get ids from pre-trained model vocab
    train_code, train_mask = convert_examples_to_features(train_source_code, tokenizer)
    test_code, test_mask = convert_examples_to_features(test_source_code, tokenizer)
    val_code, val_mask = convert_examples_to_features(val_source_code, tokenizer)

    summary_vocab_size = data["config"]["comvocabsize"]
    code_vocab_size = data["config"]["datvocabsize"]

    train_sbt = None
    val_sbt = None
    test_sbt = None
    sbt_vocab_size = -1
    if cf.enable_sbt:
        train_sbt = data["strain"]
        val_sbt = data["sval"]
        test_sbt = data["stest"]

        # sbt_token_dict = data["smltok"]
        # sbt_token_i2w = sbt_token_dict["i2w"]

        sbt_vocab_size = data["config"]["smlvocabsize"]

    train_func = None
    val_func = None
    test_func = None
    if cf.enable_func:
        train_func = data["ftrain"]
        val_func = data["fval"]
        test_func = data["ftest"]

        # sbt_token_dict = data["smltok"]
        # sbt_token_i2w = sbt_token_dict["i2w"]

        sbt_vocab_size = data["config"]["smlvocabsize"]

    summary_len = data["config"]["comlen"]

    cf.UNK_token_id = summary_vocab_size - 1

    # obtain DataLoader for iteration
    train_dataset = CodeSummaryUmlPreTrainDataset(summary=train_summary, code=train_code, mask=train_mask,
                            sbt=train_sbt, mapping={"method2uml": train_m2u, "method2class": train_m2c}, func_name=train_func)
    val_dataset = CodeSummaryUmlPreTrainDataset(summary=val_summary, code=val_code, mask=val_mask,
                            sbt=val_sbt, mapping={"method2uml": val_m2u, "method2class": val_m2c}, func_name=val_func)
    test_dataset = CodeSummaryUmlPreTrainDataset(summary=test_summary, code=test_code, mask=test_mask,
                            sbt=test_sbt, mapping={"method2uml": test_m2u, "method2class": test_m2c}, func_name=test_func)

    train_data_loader = DataLoader(train_dataset, shuffle=True, batch_size=cf.batch_size, pin_memory=True,
                                   num_workers=cf.num_subprocesses, drop_last=True)
    val_data_loader = DataLoader(val_dataset, shuffle=False, batch_size=cf.batch_size, pin_memory=True,
                                 num_workers=cf.num_subprocesses, drop_last=True)
    test_data_loader = DataLoader(test_dataset, shuffle=False, batch_size=cf.batch_size, pin_memory=True,
                                  num_workers=cf.num_subprocesses, drop_last=True)

    return train_dataset, val_dataset, test_dataset, train_data_loader, val_data_loader, test_data_loader, \
           code_vocab_size, sbt_vocab_size, summary_vocab_size, summary_vocab, summary_len, val_ids, test_ids, uml_train, uml_val, uml_test