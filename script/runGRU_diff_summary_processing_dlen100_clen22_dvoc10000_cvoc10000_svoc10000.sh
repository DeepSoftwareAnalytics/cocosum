#!/usr/bin/env bash

cd ../
mkdir -p "./output_diff_summary_processing_dlen100_clen22_dvoc10000_cvoc10000_svoc10000"
cd output_diff_summary_processing_dlen100_clen22_dvoc10000_cvoc10000_svoc10000

gpu_id="0"
#
declare -a  data_set_names=(
                            "dlen100_clen22_slen100_dvoc10000_cvoc10000_svoc10000_djl1_dfp0_dsi1_dlc1_dr0_cfp0_csi0_cfd0_ufd1_dataset" \
                            "dlen100_clen22_slen100_dvoc10000_cvoc10000_svoc10000_djl1_dfp0_dsi1_dlc1_dr0_cfp0_csi0_cfd1_ufd1_dataset" \
                            "dlen100_clen22_slen100_dvoc10000_cvoc10000_svoc10000_djl1_dfp0_dsi1_dlc1_dr0_cfp0_csi1_cfd0_ufd1_dataset" \
                            "dlen100_clen22_slen100_dvoc10000_cvoc10000_svoc10000_djl1_dfp0_dsi1_dlc1_dr0_cfp0_csi1_cfd1_ufd1_dataset" \
                            "dlen100_clen22_slen100_dvoc10000_cvoc10000_svoc10000_djl1_dfp0_dsi1_dlc1_dr0_cfp1_csi0_cfd0_ufd1_dataset" \
                            "dlen100_clen22_slen100_dvoc10000_cvoc10000_svoc10000_djl1_dfp0_dsi1_dlc1_dr0_cfp1_csi0_cfd1_ufd1_dataset" \
                            "dlen100_clen22_slen100_dvoc10000_cvoc10000_svoc10000_djl1_dfp0_dsi1_dlc1_dr0_cfp1_csi1_cfd0_ufd1_dataset" \
                            "dlen100_clen22_slen100_dvoc10000_cvoc10000_svoc10000_djl1_dfp0_dsi1_dlc1_dr0_cfp1_csi1_cfd1_ufd1_dataset" \
                            )

# "/mnt/Data/csn/dataset/diff_summary_processing/dlen100_clen22_slen100_dvoc10000_cvoc10000_svoc10000_djl0_dfp0_dsi0_dlc0_dr0_cfp1_csi1_cfd1_ufd1_dataset.pkl"\
declare -a  data_set_paths=(
"/mnt/Data/csn/dataset/diff_summary_processing/dlen100_clen22_slen100_dvoc10000_cvoc10000_svoc10000_djl1_dfp0_dsi1_dlc1_dr0_cfp0_csi0_cfd0_ufd1_dataset.pkl" \
"/mnt/Data/csn/dataset/diff_summary_processing/dlen100_clen22_slen100_dvoc10000_cvoc10000_svoc10000_djl1_dfp0_dsi1_dlc1_dr0_cfp0_csi0_cfd1_ufd1_dataset.pkl" \
"/mnt/Data/csn/dataset/diff_summary_processing/dlen100_clen22_slen100_dvoc10000_cvoc10000_svoc10000_djl1_dfp0_dsi1_dlc1_dr0_cfp0_csi1_cfd0_ufd1_dataset.pkl" \
"/mnt/Data/csn/dataset/diff_summary_processing/dlen100_clen22_slen100_dvoc10000_cvoc10000_svoc10000_djl1_dfp0_dsi1_dlc1_dr0_cfp0_csi1_cfd1_ufd1_dataset.pkl" \
"/mnt/Data/csn/dataset/diff_summary_processing/dlen100_clen22_slen100_dvoc10000_cvoc10000_svoc10000_djl1_dfp0_dsi1_dlc1_dr0_cfp1_csi0_cfd0_ufd1_dataset.pkl" \
"/mnt/Data/csn/dataset/diff_summary_processing/dlen100_clen22_slen100_dvoc10000_cvoc10000_svoc10000_djl1_dfp0_dsi1_dlc1_dr0_cfp1_csi0_cfd1_ufd1_dataset.pkl" \
"/mnt/Data/csn/dataset/diff_summary_processing/dlen100_clen22_slen100_dvoc10000_cvoc10000_svoc10000_djl1_dfp0_dsi1_dlc1_dr0_cfp1_csi1_cfd0_ufd1_dataset.pkl" \
"/mnt/Data/csn/dataset/diff_summary_processing/dlen100_clen22_slen100_dvoc10000_cvoc10000_svoc10000_djl1_dfp0_dsi1_dlc1_dr0_cfp1_csi1_cfd1_ufd1_dataset.pkl" \
          )

#declare -a  data_set_names=("dlen100_clen22_slen100_dvoc10000_cvoc10000_svoc10000_djl0_dfp0_dsi0_dlc0_dr0_cfp1_csi1_cfd1_ufd1_dataset"\
#                        )
#declare -a  data_set_paths=( "/mnt/Data/csn/dataset/diff_summary_processing/dlen100_clen22_slen100_dvoc10000_cvoc10000_svoc10000_djl0_dfp0_dsi0_dlc0_dr0_cfp1_csi1_cfd1_ufd1_dataset.pkl"\
# )
numOfDataSets=${#data_set_names[@]}

batch_size="256"

modeltype='att-gru'

code_dim="128"
summary_dim="128"
sbt_dim="0"
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
