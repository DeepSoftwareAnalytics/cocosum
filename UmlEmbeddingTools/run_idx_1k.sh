#########################################################################
# File Name: run_idx.sh
# Author: Lun Du
# Created Time: Sun 08 Mar 2020 07:30:29 PM UTC
#########################################################################
#!/bin/bash

python method2class.py --uml 1k/valid/umls.pkl --index 1k/valid/method2uml_index.pkl --method 1k/valid/methods.pkl --output data/1k/valid

python method2class.py --uml 1k/train/umls.pkl --index 1k/train/method2uml_index.pkl --method 1k/train/methods.pkl --output data/1k/train

python method2class.py --uml 1k/test/umls.pkl --index 1k/test/method2uml_index.pkl --method 1k/test/methods.pkl --output data/1k/test

python integrate.py --dataset 1k/dataset.pkl --uml_home 1k/ --output data/1k 
        
