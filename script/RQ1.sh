#!/usr/bin/env bash
# What is the effectiveness of our proposed modelCoCoGUM?
# codenn
python train.py -modeltype='codenn' -root=data/csn/ -gpu_count=1
# h-deepcom
python train.py -modeltype='h-deepcom' -root=data/csn/ -gpu_count=1
# att-gru
python train.py -modeltype='att-gru' -root=data/csn/ -gpu_count=1
# ast-att-gru
python train.py -modeltype='ast-att-gru' -root=data/csn/ -gpu_count=1
# uml
python train.py -modeltype='uml' -root=data/csn/ -gpu_count=1 -dty="ASTAttGRU_AttTwoChannelTrans"
