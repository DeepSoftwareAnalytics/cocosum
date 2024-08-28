#!/usr/bin/env bash

cd ../
mkdir -p "./output_difflen"
cd output_difflen

gpu_id="0"

declare -a  data_set_names=("csn_code_nofilter_d100c13s100_vocd1000c1000s1000" )
#                            "csn_code_nofilter_d100c13s100_vocd1000c2000s1000" )

#declare -a  data_set_paths=("../data/my_sbt_all_voc1000/dataset.pkl" \
#                            "../data/my_sbt_all/dataset.pkl" \
#                            "../data/leclair/standard_data.pkl")
declare -a  data_set_paths=("/mnt/Data/csn/diffvocab_datasets_with_sbt/csn_code_nofilter_d100c13s100_vocd1000c1000s1000.pkl" )
#                            "/mnt/Data/csn/diffvocab_datasets_with_sbt/csn_code_nofilter_d100c13s100_vocd1000c2000s1000.pkl")
numOfDataSets=${#data_set_names[@]}

batch_size="256"

modeltype='att-gru'

code_dim="128"
summary_dim="128"
sbt_dim="0"

rnn_hidden_size="128"

learning_rate="0.00005"


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
                                out_path="./${data_set_names[$index]}-$mt-$bs-$code_tok_ed-$sum_tok_ed-$sbt_tok_ed-$rhs-$lr-$current_time.pth"
#                               out_path=""
                               echo $out_path

                               python ../runGRU.py -gpu_id "$gpu_id" -data "${data_set_paths[$index]}" -modeltype "$mt" \
                               -batch_size "$bs" -code_dim "$code_tok_ed" -summary_dim "$sum_tok_ed" -sbt_dim "$sbt_tok_ed" \
                               -rnn_hidden_size "$rhs" -lr "$lr" -out_path "$out_path" | tee "$log_path"

                           done
                       done
                   done
               done
           done
       done
   done
done
