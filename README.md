# CoCoSUM 

##  1. Environment 

OS: Ubuntu 16.04.6 LTS / 18.04.4 LTS 

    $ conda create -n CoCoSUM python=3.6
    $ conda activate CoCoSUM
    $ conda install numpy pytorch-gpu=1.3.1
    $ pip install torch-scatter==1.4.0  
    $ pip install torch-sparse==0.4.3  
    $ pip install torch-cluster==1.4.5 
    $ pip install torch-geometric==1.3.2
    $ pip install git+https://github.com/casics/spiral.git
    $ pip install nltk==3.2.5  ipdb==0.13.3 javalang==0.12.0 networkx==2.3 BeautifulSoup4
    $ pip install lxml
    $ conda install natsort==7.0.1 pydot==1.4.1 
    $ pip install tensorflow-gpu==2.0.0 tensorflow-hub==0.7.0 decamelize==0.1.2 
    $ sudo apt install tofrodos


Docker

```
docker pull enshi/cocosum:v1
docker run --runtime=nvidia -itd --name=cocosum_env --ipc=host --net=host -v $HOME/sciteamdrive2:/sciteamdrive2 enshi/cocosum:v0
# the above command will return the  [container id]
docker start  [container id]
docker exec -it   [container id] /bin/bash
```

you should download [srcml 0.9.5](http://131.123.42.38/lmcrs/beta/) and install it using:

    $ sudo dpkg -i srcML-Ubuntu[your version]

## 2. Quick start
It will  spend so many time in data processing. Thus, we provide two version open source code: all code based and mini data based.
### All data

#### From scratch 

If all the requirements are met, just run: 
    
    $ fromdos  ./pipeline.sh
    $ bash ./pipeline.sh
    
It will 
* download the [Code Search Net Raw Data](https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/java.zip) and save in data/csn/java.
* get code, summary, sbt ,and uml & save in csn/data/code, csn/data/summary, csn/data/sbt, and  csn/data/uml respectively. 
* Build vocabulary and dataset &  save  in  csn/data/raw_vocabulary  and csn/data/dataset
* train and evaluate the model.

More details you can see [here](datapreprocess/CodeSearchNet/readme.md)

#### Obtain Dataset and Train a Model

##### step 1: Data Preparation
It will  spend so many time in full data processing. Thus, we provide a processed dataset.
you can:

    $ wget http://157.55.87.47:13111/data.tar.bz2
    $ tar -jxvf data.tar.bz2
    $ mv data/csn/* CodeSum/data/csn/
    $ cd ./CodeSum/data/csn/
    $ cp dlen100_clen12_slen435_dvoc10000_cvoc10000_svoc10000_dataset.pkl  ./dataset
    $ cp cfp1_csi1_cfd0_clc1.pkl ./summary
    $ cd ../../..
    
#####  step 2: Training a model
You can either load a trained model from ./model/model.pth, or train it from scratch by:

    $ cd CodeSum
    $ python runGruUml.py -modeltype=uml  -dty=ASTAttGRU_AttTwoChannelTrans -dataset_path "./data/csn/"
##### step 3: Automatic Evaluation
Please specify model_file and prediction_path according to your file's location

    $ python evaluate.py -modeltype=uml -model_root_path="./model/" -model_file='model.pth' -prediction_path='./predict' 
    
### Mini data

Similar to All data.
If all the requirements are met, just run: 

    $ cd CodeSum
    $ fromdos  ./pipeline_mini_dataset.sh
    $ bash ./pipeline_mini_dataset.sh

It will take about 5 min.
* The log of data processing will save in  ./datapreprocess/CodeSearchNet/log
* The  model will save in ./model
* we provide four metrics: BlEU-4,  ROUGe, Cider, and Meteor.

### [Optional] data format: pickle2jsonl

We provide a script for data format transformation from pickle to json/jsonl.

```
cd pickle2jsonl
python pickle2jsonl.py --filename {PATH_TO_PICKLE_FILE} --outdir {JSON_OUTPUT_DIR}
```

### RQ1
```
bash script/RQ1.sh
```
If you encounter problems running the script, do `fromdos script/RQ1.sh` and try again.
 
### RQ2
```
bash script/RQ2.sh
```
    
### RQ3
See jupyter notebook [here](RQ/compare_diff_codelen_sumlen.ipynb).
    
### RQ4
```
bash script/RQ4.sh
```  
 
