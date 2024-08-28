import torch
from torch.utils.data import Dataset


class CodeSummaryDataset(Dataset):
    def __init__(self, summary, code, sbt=None):
        super(CodeSummaryDataset, self).__init__()

        self.seq_num = len(summary)

        assert (len(code) == self.seq_num)

        self.summary = []
        self.code = []

        if sbt is not None:
            assert (len(sbt) == self.seq_num)
            self.sbt = []
        else:
            self.sbt = None

        for key, seq in summary.items():
            self.summary.append(seq)
            self.code.append(code[key])
            if sbt is not None:
                self.sbt.append(sbt[key])

        # Use only when summary, code and sbt are padded already
        # and each sequence has the same length
        self.summary = torch.LongTensor(self.summary)
        self.code = torch.LongTensor(self.code)

        if self.sbt is not None:
            self.sbt = torch.LongTensor(self.sbt)

    def __len__(self):
        return self.seq_num

    def __getitem__(self, idx):

        if self.sbt is not None:
            return self.code[idx], self.sbt[idx], self.summary[idx]
        else:
            return self.code[idx], self.summary[idx]


class CodeSummaryUmlDataset(Dataset):
    def __init__(self, summary, code, sbt=None, mapping=None, func_name=None):
        super(CodeSummaryUmlDataset, self).__init__()

        self.seq_num = len(summary)

        assert (len(code) == self.seq_num)

        self.summary = []
        self.code = []

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
            if sbt is not None:
                self.sbt.append(sbt[key])
            if mapping is not None:
                self.method2uml.append(mapping["method2uml"][key])
                self.method2class.append(mapping["method2class"][key])
            if func_name is not None:
                self.func_name.append(func_name[key])
        
        self.summary = torch.LongTensor(self.summary)
        self.code = torch.LongTensor(self.code)

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
            return self.code[idx], self.sbt[idx], self.func_name[idx], self.summary[idx], self.method2uml[idx], self.method2class[idx]
        elif self.sbt is not None and self.method2uml is not None:
            return self.code[idx], self.sbt[idx], self.summary[idx], self.method2uml[idx], self.method2class[idx]
        elif self.sbt is not None and self.func_name is not None:
            return self.code[idx], self.sbt[idx], self.func_name[idx], self.summary[idx]
        elif self.method2uml is not None and self.func_name is not None:
            return self.code[idx], self.func_name[idx], self.summary[idx], self.method2uml[idx], self.method2class[idx]
        elif self.sbt is not None:
            return self.code[idx], self.sbt[idx], self.summary[idx]
        elif self.method2uml is not None:
            return self.code[idx], self.summary[idx], self.method2uml[idx], self.method2class[idx]
        elif self.func_name is not None:
            return self.code[idx], self.func_name[idx], self.summary[idx]
        else:
            return self.code[idx], self.summary[idx]