# preprocess
* debug in 157.55.183.144 (12 cpu cores)

### Step 0: Download original csn (CodeSearchNet) data.

    mkdir -p data/csn
    cd data/csn
    wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/java.zip
    unzip -o java.zip
    
### Step 1: Download git repository

    cd datapreprocess/CodeSearchNet/
    python 1_get_git_repository.py
    
        [DEBUG] processing /mnt/enshi/CodeSum/data/csn/java/final/jsonl/train/java_train_10.jsonl.gz
        [DEBUG] processing /mnt/enshi/CodeSum/data/csn/java/final/jsonl/train/java_train_14.jsonl.gz
        write file to /mnt/enshi/CodeSum/data/csn/github_repositories/train/sha2url.pkl
        write file to /mnt/enshi/CodeSum/data/csn/github_repositories/train/sha2repoName.pkl
        [DEBUG] train 00:12:50
        [DEBUG] processing /mnt/enshi/CodeSum/data/csn/java/final/jsonl/valid/java_valid_0.jsonl.gz
        write file to /mnt/enshi/CodeSum/data/csn/github_repositories/valid/sha2url.pkl
        write file to /mnt/enshi/CodeSum/data/csn/github_repositories/valid/sha2repoName.pkl
        [DEBUG] valid 00:00:38
        [DEBUG] processing /mnt/enshi/CodeSum/data/csn/java/final/jsonl/test/java_test_0.jsonl.gz
        write file to /mnt/enshi/CodeSum/data/csn/github_repositories/test/sha2url.pkl
        write file to /mnt/enshi/CodeSum/data/csn/github_repositories/test/sha2repoName.pkl
        [DEBUG] test 00:00:54

        
### Step 2: Add fid to original csn data.
        
    cd datapreprocess/CodeSearchNet/
    python 2_add_id.py
    
        train
        ../../data/csn/java/final/jsonl/train/java_train_0.jsonl.gz
        ....
        ../../data/csn/java/final/jsonl/train/java_train_15.jsonl.gz
        [DEBUG] valid
        [DEBUG] ../../data/csn/java/final/jsonl/valid/java_valid_0.jsonl.gz
        test
        ../../data/csn/java/final/jsonl/test/java_test_0.jsonl.gz
        write file to ../../data/csn/csn.pkl
        [DEBUG]  time cost :00:00:55
        
*  diff  in key : ["val"] -> ["valid"]


### Step 3: Token preprocessing for code, summary, and sbt.
        
    cd datapreprocess/CodeSearchNet/
    sudo dpkg -i srcML-Ubuntu18.04.deb or sudo dpkg -i srcML-Ubuntu14.04.deb
    python 3_token_preprocessing.py
    
        [DEBUG] Generate SBT
        write file to ../../data/csn/sbt/sbts.pkl
        write file to ../../data/csn/sbt/sbts_tokens.pkl
        [DEBUG] time cost 03:28:06
        [DEBUG] Process code
        write file to ../../data/csn/code/djl1_dfp0_dsi1_dlc1_dr1.pkl
        [DEBUG] time cost 00:12:54
        [DEBUG] Process summary
        write file to ../../data/csn/summary/cfp1_csi1_cfd0_clc1.pkl
        [DEBUG] time cost 00:03:04
        [DEBUG] time cost 03:44:37

* sbt: diff  ["'parameter_list'"] -> ["'parameter_list_()'"]
* code_token: diff. now: replace <num> and <string> -> split ID -> lowercase
                    old: split ID -> replace <num> and <string> -> lowercase
* summary: diff. now: filter punctuation -> split ID -> lowercase
                    old:  lowercase ->  filter punctuation -> split ID 

### Step 4: Extract uml
#### Step 4.1: Split large dataset to a set of small datasets
    cd datapreprocess/CodeSearchNet/
    python 4_1_split_dataset.py
    
        [DEBUG] Each small data size: 3000 
        write file to ../../data/csn/split_data/train/0_3000.pkl
        ...
        write file to ../../data/csn/split_data/train/453000_454451.pkl
        [DEBUG] Time cost 00:00:29
        write file to ../../data/csn/split_data/valid/0_3000.pkl
        ...
        write file to ../../data/csn/split_data/valid/15000_15328.pkl
        [DEBUG] Time cost 00:00:00
        write file to ../../data/csn/split_data/test/0_3000.pkl
        ...
        write file to ../../data/csn/split_data/test/24000_26909.pkl
        [DEBUG] Time cost 00:00:00
#### Step 4.2: Generate dot using UmlGraph
    python 4_2_get_dot_with_umlgraph.py 
        NOTE: 40 cores
        [DEBUG] files count 152
        [DEBUG] 0_3000.pkl
        ...
        [DEBUG] 111000_114000.pkl  time cost : 01:38:30 
        write file to ../../data/csn/uml/m2uid/train/111000_114000_m2uid.pkl
        [DEBUG] Time cost 13:00:00
        
 * note: you can find all dot and  _*m2uid files in sciteamdrive2/v-ensh/CoCoGUM/CodeSum/data/csn/uml/
 
    
#### Step 4.3: Parse dot and get uml 
    python 4_3_get_uml.py 
        write file to ../../data/csn/uml/train/umls.pkl
        write file to ../../data/csn/uml/train/big_graph_id.pkl
        write file to ../../data/csn/uml/test/umls.pkl
        write file to ../../data/csn/uml/test/big_graph_id.pkl
        write file to ../../data/csn/uml/valid/umls.pkl
        write file to ../../data/csn/uml/valid/big_graph_id.pkl
        [DEBUG] Time cost 02:51:34
     


#### Step 4.4: Merge all small datasets

    python 4_4_merge_m2uid.py
        write file to ../../data/csn/uml/train/m2uid.pkl
        write file to ../../data/csn/uml/valid/m2uid.pkl
        write file to ../../data/csn/uml/test/m2uid.pkl
        [DEBUG] time cost 00:00:00
#### Step 4.5: Filter umls
(filter the uml  which method's class is not in)

    python 4_5_filter_class_not_in_uml.py
        [DEBUG] train
        train: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 439736/439736 [00:44<00:00, 9917.05it/s]
        [DEBUG] length of methods: 454451
        [DEBUG] length of method_class_in_package: 366712
        write file to ../../data/csn/uml/train/methods.pkl
        [DEBUG] valid
        valid: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 15208/15208 [00:01<00:00, 13728.98it/s]
        [DEBUG] length of methods: 15328
        [DEBUG] length of method_class_in_package: 12656
        write file to ../../data/csn/uml/valid/methods.pkl
        [DEBUG] test
        test: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 26867/26867 [00:02<00:00, 9553.86it/s]
        [DEBUG] length of methods: 26909
        [DEBUG] length of method_class_in_package: 25448
        write file to ../../data/csn/uml/test/methods.pkl
        write file to ../../data/csn/uml/correct_uml_fid.pkl
        [DEBUG] time cost 00:03:37

        
#### Step 5: Uml embedding
    cd datapreprocess/CodeSearchNet/
    python 5_uml_embedding.py
        [DEBUG] [...] Start loading embedding model.
        INFO:absl:Using /tmp/tfhub_modules to cache modules.
        2020-10-13 14:53:20.681337: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1
        2020-10-13 14:53:20.682466: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 0 with properties:
        name: Tesla V100-PCIE-16GB major: 7 minor: 0 memoryClockRate(GHz): 1.38
        pciBusID: 0001:00:00.0
        2020-10-13 14:53:20.683748: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 1 with properties:
        name: Tesla V100-PCIE-16GB major: 7 minor: 0 memoryClockRate(GHz): 1.38
        pciBusID: 0002:00:00.0
        2020-10-13 14:53:20.683815: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.0
        ...
        [...] Loading
        [X] Loaded.
        [...] Saving
        [X] Saved
        [DEBUG] time cost 01:04:51
    
#### Step 6:  Get correct fid
    cd datapreprocess/CodeSearchNet/
    python 6_get_correct_id.py 
        trainlen final_correct_id  303420
        validlen final_correct_id  9575
        testlen final_correct_id  21227
        write file to ../../data/csn/correct_fid/cfd1_sfd1_ufd1_fid.pkl
        time cost 00:00:16

#### Step 7: word count and build vocabulary
    cd datapreprocess/CodeSearchNet/
    python 7_build_vocabulary.py 
        write file to ../../data/csn/vocab_raw/summary_word_count_cfp1_csi1_cfd0_clc1.pkl
        write file to ../../data/csn/vocab_raw/code_word_count_djl1_dfp0_dsi1_dlc1_dr1.pkl
        write file to ../../data/csn/vocab_raw/sbt_word_count.pkl
        save json in ../../data/csn/vocab_raw/csn_trainingset_djl1_dfp0_dsi1_dlc1_dr1_cfp1_csi1_cfd0_clc1.json
        [DEBUG] time cost 00:02:52

#### Step 8: build dataset
    cd datapreprocess/CodeSearchNet/
     
    python 8_build_dataset.py -dlen 100  -clen  12 -slen 435 |tee ./log/8_build_dataset.txt
        [INFO] [Setting] DEBUG: True
        [INFO] [Setting] code_tokens_using_javalang_results: True
        [INFO] [Setting] code_filter_punctuation : False
        [INFO] [Setting] code_split_identifier: True
        [INFO] [Setting] code_lower_case: True
        [INFO] [Setting] code_replace_string_num: True
        [INFO] [Setting] summary_filter_punctuation: True
        [INFO] [Setting] summary_split_identifier: True
        [INFO] [Setting] summary_filter_bad_cases: True
        [INFO] [Setting] code_len: 100
        [INFO] [Setting] summary_len: 12
        [INFO] [Setting] sbt_len: 435
        [INFO] [Setting] code_vocab_size: -1
        [INFO] [Setting] summary_vocab_size: -1
        [INFO] [Setting] sbt_vocab_size: -1
        [INFO] [Setting]  uml_filter_data_bad_cases: 1
        [INFO] [Setting]  save file in : ../../data/csn/dataset
        write file to ../../data/csn/dataset/dlen100_clen12_slen435_dvoc10000_cvoc10000_svoc10000_dataset.pkl
        [DEBUG] time cost 00:01:28


#### Step 9: train
    cd ./
    python runGruUml.py -modeltype=uml  -dty=ASTAttGRU_AttTwoChannelTrans |tee train.txt
    INFO] [Setting] EXP: True
        [INFO] [Setting] DEBUG: False
        [INFO] [Setting] trim_til_EOS: True
        [INFO] [Setting] use_full_sum: True
        [INFO] [Setting] use_oov_sum: False
        [INFO] [Setting] Method: uml
        [INFO] [Setting] in_path: ./data/csn/dataset/dlen100_clen12_slen435_dvoc10000_cvoc10000_svoc10000_dataset.pkl
        [INFO] [Setting] GPU id: 1
        [INFO] [Setting] num_epochs: 40
        [INFO] [Setting] batch_size: 256
        [INFO] [Setting] code_dim: 128
        [INFO] [Setting] sbt_dim: 128
        [INFO] [Setting] summary_dim: 128
        [INFO] [Setting] rnn_hidden_size: 256
        [INFO] [Setting] lr: 0.001000
        [INFO] [Setting] num_subprocesses: 4
        [INFO] [Setting] eval_frequency: 1
        [INFO] [Setting] out_name: mtuml_bs256_ddim128_cdim128_sdim128_hdim256_lr0.001_gtyHGCN_gdp0.6_0.6_0.6-gfn0-gpp0_gfc1_wd0.3_gss0_gnf513_gnh128_256-aggMean-cloFalse_dtyASTAttGRU_AttTwoChannelTrans20201014053548.pt
        [INFO] [Setting] decoder_type: ASTAttGRU_AttTwoChannelTrans
        [INFO] [Setting] gnn_num_features: 513
        [INFO] [Setting] gnn_num_hidden: [128, 256]
        [INFO] [Setting] gnn_dropout: [0.6, 0.6, 0.6]
        [INFO] [Setting] enable_sbt: True
        [INFO] [Setting] gnn_type:HGCN
        [INFO] [Setting] weight_decay:0.3
        [INFO] [Setting] rnn_dropout:0
        [INFO] [Setting] rnn_num_layers:1
        [INFO] [Setting] schedual_sampling_prob:0
        [INFO] [Setting] gnn_concat:True
        [INFO] [Setting] gnn_concat_uml:False
        [INFO] [Setting] uml path:./data/csn/uml/uml_dataset.pt
        [INFO] [Setting] random_seed: 113111
        [INFO] [Setting] device: cuda:1
    ...
    
#### Step 10: evaluation  

    python evaluate.py -modeltype=uml -model_root_path="./model/" -model_file='model.pth' -prediction_path='./predict'
        INFO:CodeSummary:[Setting] device: cuda:1
        INFO:CodeSummary:[Setting] device: cuda:1
        write file to ./predict/mtcodenn_bs256_ddim128_cdim128_sdim128_hdim256_lr0.001_gtyHGCN_gdp0.6_0.6_0.6-gfn0-gpp0_gfc1_wd0.3_gss0_gnf513_gnh128_256-aggMean-cloFalse_dtyASTAttGRU_AttTwoChannelTrans20201014052651.pt.pred
        time cost:  216.03698094654828
        ROUGe:  0.12982875665012897
        Cider:  0.12475952827086477
        Meteor:  0.03971721696188747

* note: please specific the model_file like: mtcodenn_bs256_ddim128_cdim128_sdim128_hdim256_lr0.001_gtyHGCN_gdp0.6_0.6_0.6-gfn0-gpp0_gfc1_wd0.3_gss0_gnf513_gnh128_256-aggMean-cloFalse_dtyASTAttGRU_AttTwoChannelTrans20201014052651.pt

        

## Result:
###csn_length_count.py
```
read pickle data time train:  13.503300954587758
code length info: 
split_and_filter docstring_tokens time:  93.31471873261034
max_len: 3279
min_len: 0
90%: 34
80%: 22
process docstring_tokens time:  94.95786743052304

Summary length info: 
split_and_filter docstring_tokens time:  461.4989451020956
max_len: 75149
min_len: 11
90%: 185
80%: 113
process code_tokens time:  463.6031428510323
```

### csn_filter_noSBT.py
1. build vocab on first time using full lengths of code and summary,vocab_size = -1 <br>
if we need change vocab size, just choice vocab's top n where n = vocab_size

2. add parallelize model to process data, it will save a lot of time.

3. The finally result we just cost 20 min to build a new datasets.

The log on my computer is shown below:<br>
""<br>
Read train data: method-summary pair 454451,  code_token_vocab_size 50000, summary_token_vocab_size 50000  <br>
F:/msra/dataset/csn/java/final/jsonl/valid\java_valid_0.jsonl.gz<br>
Read test/valid data: method-summary pair 15328<br>
F:/msra/dataset/csn/java/final/jsonl/test\java_test_0.jsonl.gz<br>
Read test/valid data: method-summary pair 26909<br>
time:  946.9359269<br>
""

### filter_summary.py
- train data size: 454451, filtered out 55181, 0.1214
- val data size: 15328, filtered out 3153, 0.2057
- test data size: 26909, filtered out 3502, 0.1301

### TODO

- code: 原来的code是split后replace，导致字符串"hello world" -> "<string>" ,"hello", "wolrd"",已经修正。
- summary： summary是先lowercase在split, 导致camelCase 处理后为camelcase 而不是camel, case.
- sbt: parameter_list()
- uml: umlgraph 解析出来的dot文件不一定相同, 比如：459808.dot
- check exist
- save data in Teamdrive
- get exact score: 模型的运行和当时ASE时候不相同，参数和运行中总是出bug。