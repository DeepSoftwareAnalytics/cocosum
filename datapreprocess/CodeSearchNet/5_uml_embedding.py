#!/usr/bin/env python
# !-*-coding:utf-8 -*-
'''
@version: python3.*
@author: ‘v-ensh‘
@license: Apache Licence 
@contact: *******@qq.com
@site: 
@software: PyCharm
@file: 5_uml_embedding.py
@time: 10/13/2020 2:05 PM
'''
import os
import time
import sys
sys.path.append("../../")
from util.Config import Config as cf
from util.LoggerUtil import set_logger, debug_logger
from util.DataUtil import time_format
set_logger(cf.DEBUG)
start_time = time.perf_counter()
uml_dir = "../../data/csn/uml/"
parts = ["train", 'valid', 'test']
for part in parts:
    data_path = os.path.join(uml_dir, part)
    # uml2pyg
    uml_path = os.path.join(data_path, "umls.pkl")
    cmd_uml2pyg = "python uml2pyg.py --data %s --output %s" % (uml_path, data_path)
    os.system(cmd_uml2pyg)
    # method2class
    m2uid_path = os.path.join(data_path, "m2uid.pkl")
    methods_path = os.path.join(data_path, "methods.pkl")
    cmd_m2c = "python method2class.py --uml %s  --index %s --method %s --output %s" \
              % (uml_path, m2uid_path, methods_path, data_path)
    os.system(cmd_m2c)

# "python integrate.py --dataset all/dataset_with_sbt.pkl --uml_home all/ --output data/all"
integrate_cmd = "python integrate.py  --uml_home %s --output %s " % (uml_dir, uml_dir)
os.system(integrate_cmd)
debug_logger("time cost %s" % time_format(time.perf_counter() - start_time))

# os.system("python integrate.py --dataset all/dataset_with_sbt.pkl --uml_home all/ --output data/all")
# os.system("python uml2pyg.py --data /mnt/enshi/CodeSum/UmlEmbeddingTools/data/1k/valid/ --output /mnt/enshi/CodeSum/UmlEmbeddingTools/res/valid")
# os.system("")
# os.system("")
# # method2class
# os.system("python method2class.py --uml all/valid/umls.pkl --index all/valid/method2uml_index.pkl --method all/valid/method_class_in_package.pkl --output data/all/valid")
# os.system("python method2class.py --uml all/train/umls.pkl --index all/train/method2uml_index.pkl --method all/train/method_class_in_package.pkl --output data/all/train")
# os.system("python method2class.py --uml all/test/umls.pkl --index all/test/method2uml_index.pkl --method all/test/method_class_in_package.pkl --output data/all/test")

# merge
# os.system("python integrate.py --dataset all/dataset_with_sbt.pkl --uml_home all/ --output data/all")