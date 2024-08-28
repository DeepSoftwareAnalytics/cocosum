#!/usr/bin/env bash

cd ../
mkdir -p "./output_diff_svoc_dvoc10000_cvoc10000_djl1_dfp0_dsi1_dlc1_dr1_cfp1_csi1_cfd1_ufd1"
cd output_diff_svoc_dvoc10000_cvoc10000_djl1_dfp0_dsi1_dlc1_dr1_cfp1_csi1_cfd1_ufd1

gpu_id="0"
declare -a  data_set_names=('dlen135_clen16_slen600_dvoc10000_cvoc10000_svoc1000_djl1_dfp0_dsi1_dlc1_dr1_cfp1_csi1_cfd1_ufd1_dataset.pkl' \
 'dlen135_clen16_slen600_dvoc10000_cvoc10000_svoc5000_djl1_dfp0_dsi1_dlc1_dr1_cfp1_csi1_cfd1_ufd1_dataset.pkl' \
 'dlen135_clen16_slen600_dvoc10000_cvoc10000_svoc10000_djl1_dfp0_dsi1_dlc1_dr1_cfp1_csi1_cfd1_ufd1_dataset.pkl' \
 'dlen135_clen16_slen600_dvoc10000_cvoc10000_svoc15000_djl1_dfp0_dsi1_dlc1_dr1_cfp1_csi1_cfd1_ufd1_dataset.pkl' \
 'dlen135_clen16_slen600_dvoc10000_cvoc10000_svoc20000_djl1_dfp0_dsi1_dlc1_dr1_cfp1_csi1_cfd1_ufd1_dataset.pkl' \
 'dlen135_clen16_slen600_dvoc10000_cvoc10000_svoc25000_djl1_dfp0_dsi1_dlc1_dr1_cfp1_csi1_cfd1_ufd1_dataset.pkl' \
 'dlen135_clen16_slen600_dvoc10000_cvoc10000_svoc30000_djl1_dfp0_dsi1_dlc1_dr1_cfp1_csi1_cfd1_ufd1_dataset.pkl' \
 'dlen135_clen16_slen600_dvoc10000_cvoc10000_svoc35000_djl1_dfp0_dsi1_dlc1_dr1_cfp1_csi1_cfd1_ufd1_dataset.pkl' \
 'dlen135_clen16_slen600_dvoc10000_cvoc10000_svoc40000_djl1_dfp0_dsi1_dlc1_dr1_cfp1_csi1_cfd1_ufd1_dataset.pkl' \
 'dlen135_clen16_slen600_dvoc10000_cvoc10000_svoc45000_djl1_dfp0_dsi1_dlc1_dr1_cfp1_csi1_cfd1_ufd1_dataset.pkl' \
 'dlen135_clen16_slen600_dvoc10000_cvoc10000_svoc50000_djl1_dfp0_dsi1_dlc1_dr1_cfp1_csi1_cfd1_ufd1_dataset.pkl' \
)
# "/mnt/Data/csn/dataset/diff_code_len/dlen100_clen22_slen750_dvoc10000_cvoc10000_svoc10000_djl0_dfp0_dsi0_dlc0_dr0_cfp1_csi1_cfd1_ufd1_dataset.pkl"\
declare -a  data_set_paths=(
'/mnt/Data/csn/dataset/diff_svoc_size/dlen135_clen16_slen600_dvoc10000_cvoc10000_svoc1000_djl1_dfp0_dsi1_dlc1_dr1_cfp1_csi1_cfd1_ufd1_dataset.pkl' \
 '/mnt/Data/csn/dataset/diff_svoc_size/dlen135_clen16_slen600_dvoc10000_cvoc10000_svoc5000_djl1_dfp0_dsi1_dlc1_dr1_cfp1_csi1_cfd1_ufd1_dataset.pkl' \
 '/mnt/Data/csn/dataset/diff_svoc_size/dlen135_clen16_slen600_dvoc10000_cvoc10000_svoc10000_djl1_dfp0_dsi1_dlc1_dr1_cfp1_csi1_cfd1_ufd1_dataset.pkl' \
 '/mnt/Data/csn/dataset/diff_svoc_size/dlen135_clen16_slen600_dvoc10000_cvoc10000_svoc15000_djl1_dfp0_dsi1_dlc1_dr1_cfp1_csi1_cfd1_ufd1_dataset.pkl' \
 '/mnt/Data/csn/dataset/diff_svoc_size/dlen135_clen16_slen600_dvoc10000_cvoc10000_svoc20000_djl1_dfp0_dsi1_dlc1_dr1_cfp1_csi1_cfd1_ufd1_dataset.pkl' \
 '/mnt/Data/csn/dataset/diff_svoc_size/dlen135_clen16_slen600_dvoc10000_cvoc10000_svoc25000_djl1_dfp0_dsi1_dlc1_dr1_cfp1_csi1_cfd1_ufd1_dataset.pkl' \
 '/mnt/Data/csn/dataset/diff_svoc_size/dlen135_clen16_slen600_dvoc10000_cvoc10000_svoc30000_djl1_dfp0_dsi1_dlc1_dr1_cfp1_csi1_cfd1_ufd1_dataset.pkl' \
 '/mnt/Data/csn/dataset/diff_svoc_size/dlen135_clen16_slen600_dvoc10000_cvoc10000_svoc35000_djl1_dfp0_dsi1_dlc1_dr1_cfp1_csi1_cfd1_ufd1_dataset.pkl' \
 '/mnt/Data/csn/dataset/diff_svoc_size/dlen135_clen16_slen600_dvoc10000_cvoc10000_svoc40000_djl1_dfp0_dsi1_dlc1_dr1_cfp1_csi1_cfd1_ufd1_dataset.pkl' \
 '/mnt/Data/csn/dataset/diff_svoc_size/dlen135_clen16_slen600_dvoc10000_cvoc10000_svoc45000_djl1_dfp0_dsi1_dlc1_dr1_cfp1_csi1_cfd1_ufd1_dataset.pkl' \
 '/mnt/Data/csn/dataset/diff_svoc_size/dlen135_clen16_slen600_dvoc10000_cvoc10000_svoc50000_djl1_dfp0_dsi1_dlc1_dr1_cfp1_csi1_cfd1_ufd1_dataset.pkl' \
 )

#declare -a  data_set_names=("dlen100_clen21_slen760_dvoc10000_cvoc10000_svoc10000_djl1_dfp0_dsi1_dlc1_dr1_cfp1_csi1_cfd1_ufd1_dataset"\
#                        )
#declare -a  data_set_paths=( "/mnt/Data/csn/dataset/diff_code_len/dlen100_clen21_slen760_dvoc10000_cvoc10000_svoc10000_djl1_dfp0_dsi1_dlc1_dr1_cfp1_csi1_cfd1_ufd1_dataset.pkl"\
# )
numOfDataSets=${#data_set_names[@]}

batch_size="256"

modeltype='ast-att-gru'

code_dim="128"
summary_dim="128"
sbt_dim="128"
epoch="20"
rnn_hidden_size="128"

learning_rate="0.001"


for (( index=0; index < numOfDataSets; index++ )); do
   for mt in $modeltype; do
       for bs in $batch_size; do
           for code_tok_ed in $code_dim; do
               for sum_tok_ed in $summary_dim; do
                   for sbt_tok_ed in $sbt_dim; do
                       for rhs in $rnn_hidden_size; do
                           for lr in $learning_rate; do

                               current_time=$(date "+%Y%m%d%H%M%S")

                               log_path="./${data_set_names[$index]}-$mt-$bs-$code_tok_ed-$sum_tok_ed-$sbt_tok_ed-$rhs-$lr-$current_time.log"
                               echo $log_path

                               # Model file occupies a large space. Be careful if you want to output all the model files.
                                out_path="${data_set_names[$index]}-$mt-$bs-$code_tok_ed-$sum_tok_ed-$sbt_tok_ed-$rhs-$lr-$current_time.pth"
#                               out_path=""
                               echo $out_path

                               python ../runSeqModel.py -gpu_id "$gpu_id" -data "${data_set_paths[$index]}" -modeltype "$mt" \
                               -batch_size "$bs" -code_dim "$code_tok_ed" -summary_dim "$sum_tok_ed" -sbt_dim "$sbt_tok_ed" \
                               -rnn_hidden_size "$rhs" -lr "$lr" -epoch "$epoch" -out_path "$out_path" | tee "$log_path"

                           done
                       done
                   done
               done
           done
       done
   done
done