#!/usr/bin/env bash
# RQ4: Model generality
# Code-NN with global contexts
python train.py -modeltype='uml-code-nn' -dty='ASTAttGRU_AttTwoChannelTrans' -root=data/csn/ -gpu_count=1
# H-Deepcom with global contexts
python train.py -modeltype='uml-h-deepcom' -dty='ASTAttGRU_AttTwoChannelTrans' -root=data/csn/ -gpu_count=1
# Transformer without global contexts
python train.py -modeltype='ast-att-transformer' -dty='ASTAttGRU_AttTwoChannelTrans' -root=data/csn/ -gpu_count=1
# Transformer with global contexts
python train.py -modeltype='uml-transformer' -dty='ASTAttGRU_AttTwoChannelTrans' -root=data/csn/ -gpu_count=1
