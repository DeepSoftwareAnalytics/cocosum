# -*- coding: utf-8 -*-
# @Time      : 2020-02-21 16:33
# @Author    : Eason Hu
# @Site      : 
# @File      : get_sbt_csn.py
import subprocess
import json
import time
from CodeSum.util.DataUtil import sbt_parser


def build_data(sbt_path):
    error = 0
    print('start process sbt tree')
    start = time.time()
    with open(sbt_path, 'r') as sbt_fp:
        sbts = {}
        sbt = json.load(sbt_fp)
        fids = list(sbt.keys())
        fids.sort()
        fids = fids[:100000]  # TODO:why 1000000?
        for fid in fids:
            try:
                sbt_tokens =  sbt_parser(sbt[fid])
                sbts[fid] = sbt_tokens
            except:
                error += 1
                print(error)
            if len(sbts) % 10000 == 0:
                print(len(sbts))
                print(time.time()-start)
        """
        for fid, code in sbt.items():
            try:
                sbt_tokens = process_sbt(code)
                sbts[fid] = sbt_tokens
            except:
                error += 1
                print(error)
            if len(sbts) % 10000 == 0:
                print(len(sbts))
                print(time.time()-start)
        """
    with open('sbt_data.json', 'w') as f:
        json.dump(sbts, f)
    print(error)
    print(len(sbts))


def test_get_sbt():
    # code = "public A getA ( ) { // this is a comment \n return this; \n }"
    # code = r'public A getA() { String B = \"haha\" ; }'
    code = r'protected final int initConfiguration(){\n\t\tString propFile = this.optionPropFile.getValue();\n\n\t\tthis.configuration = this.loadProperties(propFile);\n\t\tif(this.configuration==null){\n\t\t\tSystem.err.println(this.getAppName() + \": could not load configuration properties from file <\" + propFile + \">, exiting\");\n\t\t\treturn -1;\n\t\t}\n\n\t\tif(this.configuration.get(PROP_RUN_SCRIPT_NAME)==null){\n\t\t\tSystem.err.println(this.getAppName() + \": configuration does not contain key <\" + PROP_RUN_SCRIPT_NAME + \">, exiting\");\n\t\t\treturn -1;\n\t\t}\n\t\tif(this.configuration.get(PROP_RUN_CLASS)==null){\n\t\t\tSystem.err.println(this.getAppName() + \": configuration does not contain key <\" + PROP_RUN_CLASS + \">, exiting\");\n\t\t\treturn -1;\n\t\t}\n\t\tif(this.configuration.get(PROP_JAVA_CP)==null){\n\t\t\tSystem.err.println(this.getAppName() + \": configuration does not contain key <\" + PROP_JAVA_CP + \">, exiting\");\n\t\t\treturn -1;\n\t\t}\n\n\t\tSystem.out.println(this.getAppName() + \": using configuration: \");\n\t\tSystem.out.println(\"  - run script name: \" + this.configuration.get(PROP_RUN_SCRIPT_NAME));\n\t\tSystem.out.println(\"  - run class      : \" + this.configuration.get(PROP_RUN_CLASS));\n\t\tSystem.out.println(\"  - java cp        : \" + this.configuration.get(PROP_JAVA_CP));\n\t\tSystem.out.println(\"  - auto-gen reg   : \" + this.configuration.get(PROP_EXECS_AUTOGEN_REGISTERED));\n\n\t\tfor(Object key : this.configuration.keySet()){\n\t\t\tif(StringUtils.startsWith(key.toString(), PROP_JAVAPROP_START)){\n\t\t\t\tSystem.out.println(\"  - java property  : \" + key + \" = \" + this.configuration.getProperty(key.toString()));\n\t\t\t}\n\t\t}\n\t\tSystem.out.println();\n\n\t\treturn 0;\n\t}'
    sbt_parser(code)


def process_funcom_sbt():
    sbt_path = '/datadrive/yuxuan_data/funcom/funcom_processed/functions.json'
    # sbt_path = 'F:/msra/dataset/funcom_processed/functions.json'
    build_data(sbt_path)

if __name__ == '__main__':
    # process_funcom_sbt()
    test_get_sbt()