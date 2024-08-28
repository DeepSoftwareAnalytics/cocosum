#########################################################################
# File Name: run_idx.sh
# Author: Lun Du
# Created Time: Sun 08 Mar 2020 07:30:29 PM UTC
#########################################################################
#!/bin/bash

python method2class.py --uml all/valid/umls.pkl --index all/valid/method2uml_index.pkl --method all/valid/method_class_in_package.pkl --output data/all/valid

python method2class.py --uml all/train/umls.pkl --index all/train/method2uml_index.pkl --method all/train/method_class_in_package.pkl --output data/all/train

python method2class.py --uml all/test/umls.pkl --index all/test/method2uml_index.pkl --method all/test/method_class_in_package.pkl --output data/all/test

python integrate.py --dataset all/dataset_with_sbt.pkl --uml_home all/ --output data/all
        
