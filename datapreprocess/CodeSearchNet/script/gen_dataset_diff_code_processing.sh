#!/usr/bin/env bash
cd ../
dataset_path="../../../Data/csn/dataset/diff_code_processing"
## dlen100_clen22_slen100_dvoc10000_cvoc10000_svoc10000
python  build_dataset.py  -dlen 100 -clen 22 -slen 750 -dvoc 10000 -cvoc 10000 -svoc 10000 \
        -djl "False" -dfp "False" -dsi "False"  -dlc "False" -dr "False" \
        -cfp "True" -csi "True"  -cfd "True"  -dataset_path "$dataset_path"

python  build_dataset.py  -dlen 100 -clen 22 -slen 750 -dvoc 10000 -cvoc 10000 -svoc 10000 \
        -djl "True" -dfp "False" -dsi "False"  -dlc "False" -dr "False" \
        -cfp "True" -csi "True" -cfd "True"  -dataset_path "$dataset_path"

python  build_dataset.py  -dlen 100 -clen 22 -slen 750 -dvoc 10000 -cvoc 10000 -svoc 10000 \
        -djl "True" -dfp "True" -dsi "False"  -dlc "False" -dr "False" \
        -cfp "True" -csi "True" -cfd "True"  -dataset_path "$dataset_path"

python  build_dataset.py  -dlen 100 -clen 22 -slen 750 -dvoc 10000 -cvoc 10000 -svoc 10000 \
        -djl "True" -dfp "False" -dsi "True"  -dlc "False" -dr "False" \
        -cfp "True" -csi "True" -cfd "True"  -dataset_path "$dataset_path"

python  build_dataset.py  -dlen 100 -clen 22 -slen 750 -dvoc 10000 -cvoc 10000 -svoc 10000 \
        -djl "True" -dfp "False" -dsi "False"  -dlc "True" -dr "False" \
        -cfp "True" -csi "True" -cfd "True"  -dataset_path "$dataset_path"

python  build_dataset.py  -dlen 100 -clen 22 -slen 750 -dvoc 10000 -cvoc 10000 -svoc 10000 \
        -djl "True" -dfp "False" -dsi "False"  -dlc "False" -dr "True" \
        -cfp "True" -csi "True" -cfd "True"  -dataset_path "$dataset_path"

python  build_dataset.py  -dlen 100 -clen 22 -slen 750 -dvoc 10000 -cvoc 10000 -svoc 10000 \
        -djl "True" -dfp "False" -dsi "True"  -dlc "True" -dr "False" \
        -cfp "True" -csi "True" -cfd "True"  -dataset_path "$dataset_path"

python  build_dataset.py  -dlen 100 -clen 22 -slen 750 -dvoc 10000 -cvoc 10000 -svoc 10000 \
        -djl "True" -dfp "True" -dsi "True"  -dlc "True" -dr "True" \
        -cfp "True" -csi "True" -cfd "True"  -dataset_path "$dataset_path"

 dlen50_clen10_slen100_dvoc5000_cvoc5000_svoc5000

python  build_dataset.py  -dlen 50 -clen 10 -slen 350 -dvoc 5000 -cvoc 5000 -svoc 5000 \
        -djl "False" -dfp "False" -dsi "False"  -dlc "False" -dr "False" \
        -cfp "True" -csi "True" -cfd "True"  -dataset_path "$dataset_path"

python  build_dataset.py  -dlen 50 -clen 10 -slen 350 -dvoc 5000 -cvoc 5000 -svoc 5000 \
        -djl "True" -dfp "False" -dsi "False"  -dlc "False" -dr "False" \
        -cfp "True" -csi "True" -cfd "True"  -dataset_path "$dataset_path"

python  build_dataset.py  -dlen 50 -clen 10 -slen 350 -dvoc 5000 -cvoc 5000 -svoc 5000 \
        -djl "True" -dfp "True" -dsi "False"  -dlc "False" -dr "False" \
        -cfp "True" -csi "True" -cfd "True"  -dataset_path "$dataset_path"

python  build_dataset.py  -dlen 50 -clen 10 -slen 350 -dvoc 5000 -cvoc 5000 -svoc 5000 \
        -djl "True" -dfp "False" -dsi "True"  -dlc "False" -dr "False" \
        -cfp "True" -csi "True" -cfd "True"  -dataset_path "$dataset_path"

python  build_dataset.py  -dlen 50 -clen 10 -slen 350 -dvoc 5000 -cvoc 5000 -svoc 5000 \
        -djl "True" -dfp "False" -dsi "False"  -dlc "True" -dr "False" \
        -cfp "True" -csi "True" -cfd "True"  -dataset_path "$dataset_path"

python  build_dataset.py  -dlen 50 -clen 10 -slen 350 -dvoc 5000 -cvoc 5000 -svoc 5000 \
        -djl "True" -dfp "False" -dsi "False"  -dlc "False" -dr "True" \
        -cfp "True" -csi "True" -cfd "True"  -dataset_path "$dataset_path"

python  build_dataset.py  -dlen 50 -clen 10 -slen 350 -dvoc 5000 -cvoc 5000 -svoc 5000 \
        -djl "True" -dfp "False" -dsi "True"  -dlc "True" -dr "False" \
        -cfp "True" -csi "True" -cfd "True"  -dataset_path "$dataset_path"

python  build_dataset.py  -dlen 50 -clen 10 -slen 350 -dvoc 5000 -cvoc 5000 -svoc 5000 \
        -djl "True" -dfp "True" -dsi "True"  -dlc "True" -dr "True" \
        -cfp "True" -csi "True" -cfd "True"  -dataset_path "$dataset_path"
