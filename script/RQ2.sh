#!/usr/bin/env bash
# RQ2: How much do different components/techniques con-tribute?
# CoCoGUMc
python train.py -modeltype='uml' -dty='ASTAttGRU_AttTwoChannelTrans_noclassname' -root=data/csn/ -gpu_count=1
# CoCoGUMm
python train.py -modeltype='uml' -dty="ASTAttGRU_BertAtt" -root=data/csn/ -gpu_count=1
# CoCoGUMh
python train.py -modeltype='uml' -gty="HGATSimple" -root=data/csn/ -gpu_count=1
