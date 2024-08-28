#!/usr/bin/env bash
cd ../
mkdir -p "./output_uml_diff_clen4_dvoc10000_cvoc10000_svoc10000"
path="output_uml_diff_clen4_dvoc10000_cvoc10000_svoc10000"
cd $path

gpu_id="1"

# seq len
#  10 12 15 16  21 32
dlen_s="80"
clen_s="4"
slen_s="355"

# voc size
# 1k 5k 10k 15k 20k 25k 30k 35k  40k 45k 50k
dvoc_s="10000"
cvoc_s="10000"
svoc_s="10000"

# code processing
djl="True"
dfp="False"
dsi="True"
dlc="True"
dr="True"

# summary processing
cfp="True"
csi="True"
cfd="True"

dataset_path="/mnt/Data/csn/dataset_uml/diff_clen"

batch_size="256"

modeltype='uml'   #'ast-att-gru''att-gru' "uml"

code_dim="128"
summary_dim="128"
sbt_dim="128"
epoch="70"
rnn_hidden_size="128"

learning_rate="0.001"

gty="GAT"
gdp="0.6"
gfn="False"
gpp="False"
gfc="True"
wd="0.3"
gss="0.0"
gnf="513"
gnh="256"

# build dataset
for dvoc in $dvoc_s; do
    for cvoc in $cvoc_s; do
        for svoc in $svoc_s; do
            for dlen in $dlen_s; do
                for clen in $clen_s; do
                    for slen in $slen_s; do
                      cd ../datapreprocess/CodeSearchNet
                      current_time=$(date "+%Y%m%d%H%M%S")
                      build_dataset_log_path="../../${path}/dlen${dlen}clen${clen}slen${slen}dvoc${dvoc}cvoc${cvoc}svoc${svoc}-$current_time.log"
#                       = "$build_datasetd_log_path"
                      echo $build_dataset_log_path
                      python build_dataset.py  -dlen "$dlen"  -clen  "$clen" -slen "$slen" \
                                      -dvoc "$dvoc" -cvoc "$cvoc"  -svoc "$svoc" \
                                      -djl "$djl"  -dfp "$dfp" -dsi "$dsi"  -dlc "$dlc" -dr "$dr" \
                                        -cfp "$cfp" -csi "$csi"  -cfd "$cfd"  -dataset_path "$dataset_path" | tee "$build_dataset_log_path"
                      cd ../../
                      cd "$path"

                         for mt in $modeltype; do
                           for bs in $batch_size; do
                               for code_tok_ed in $code_dim; do
                                   for sum_tok_ed in $summary_dim; do
                                       for sbt_tok_ed in $sbt_dim; do
                                           for rhs in $rnn_hidden_size; do
                                               for lr in $learning_rate; do

                                                   current_time=$(date "+%Y%m%d%H%M%S")

                                                   log_path="./dlen${dlen}-clen-${clen}-slen-${slen}-dvoc-${dvoc}-cvoc-${cvoc}-svoc-${svoc}-${mt}-${bs}-${code_tok_ed}-${sum_tok_ed}-${sbt_tok_ed}-${rhs}-${lr}-${current_time}.log"
#                                                   log_path = "$log_path"
                                                   echo $log_path

                                                   python ../runGruUml.py  -dlen "$dlen"  -clen  "$clen" -slen "$slen" \
                                                        -dvoc "$dvoc" -cvoc "$cvoc"  -svoc "$svoc" \
                                                        -djl "$djl"  -dfp "$dfp" -dsi "$dsi"  -dlc "$dlc" -dr "$dr" \
                                                        -cfp "$cfp" -csi "$csi"  -cfd "$cfd"  -dataset_path "$dataset_path" \
                                                        -gpu_id "$gpu_id"   -modeltype "$mt" \
                                                        -batch_size "$bs" -code_dim "$code_tok_ed" -summary_dim "$sum_tok_ed" -sbt_dim "$sbt_tok_ed" \
                                                        -rnn_hidden_size "$rhs" -lr "$lr" -epoch "$epoch" \
                                                        -gty "$gty"  -gdp "$gdp" -gfn "$gfn" -gpp  "$gpp" -gfc  "$gfc"  \
                                                        -wd "$wd" -gss "$gss"  -gnf "$gnf" -gnh "$gnh" | tee "$log_path"

                                               done
                                           done
                                       done
                                   done
                               done
                           done
                       done
                    done
                done
            done
        done
    done
done


