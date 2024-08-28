# -*- coding: utf-8 -*-
# @Time      : 2020-02-18 11:48
# @Author    : Eason Hu
# @Site      : 
# @File      : sbt_leclair.py

import subprocess
import json

def get_sbt(code):
    code = code.replace('</unit>', '')
    code = code.replace('<', ' <')
    code = code.replace('>', ' ')
    codes = code.split(' ')
    codes = [i for i in codes if i != '']
    res = []
    flag = 0
    for i in range(len(codes)):
        if flag == 1:
            flag = 0
            continue
        word = codes[i]
        if word == '':
            continue
        if word[:2] == '</':
            word = word.replace('</', ') ')
            word = word.replace('>', '')

        elif word[:2] != '</' and word[0] == '<':
            word = word.replace('<', '( ')
            word = word.replace('>', '')
        else:
            if codes[i-1][1:] == codes[i+1][2:]:

                word = ') ' + codes[i-1].replace('<', '') + '_' + word
                flag = 1
            else:
                continue
        res.append(word)
    return ' '.join(res)



def get_xml(in_path, out_path):
    with open(in_path, 'r') as input_f:
        data = json.load(input_f)
    error = 0
    step = 0
    with open(out_path, 'w') as output_f:
        res = {}
        for fid, funs in data.items():
            try:
                xml = subprocess.getoutput('srcml -l Java -t "{}"'.format('public String toString() {\n    return connectionString;\n  }\n'))
                # res[fid] = get_sbt(xml.split('\t')[1])
                xml = xml.replace('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<unit xmlns="http://www.srcML.org/srcML/src" revision="1.0.0" language="Java">', '')
                xml = xml.replace('\t', '')
                res[fid] = get_sbt(xml)
            except:
                error += 1
                print(error)
                continue
            step += 1
            if step % 10000 == 0:
                print(step)

        json.dump(res, output_f)

if __name__ == '__main__':
    input_path = "F:/msra/dataset/funcom_processed/functions.json"
    output_path = './functions.json'
    # get_xml(input_path, output_path)
    xml = subprocess.getoutput('srcml -l Java -t "{}"'.format('public String toString() {return connectionString;}'))
    # xml = subprocess.getoutput('srcml -l Java -t "{}"'.format('public void setLocationX(float x) {\n\tPoint3D location = super.getLocation();\n\t\n\tif (location == null) {\n\t    location = new Point3D();\n\t    super.setLocation(location);\n\t}\n\t\n\tlocation.setX(x * 10.0f);\n\tnotifyListeners();\n    }\n'))
    xml2 = subprocess.getoutput('srcml -l Java -t "{}"'.format('public String toString() { return connectionString;\n  }\n'))
    # xml2 = subprocess.getoutput('srcml -l Java -t "{}"'.format('public void setLocationX(float x) {Point3D location = super.getLocation();if (location == null) {    location = new Point3D();   super.setLocation(location);\n\t}location.setX(x * 10.0f);notifyListeners();\n    }\n'))
    xml = xml.replace(
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<unit xmlns="http://www.srcML.org/srcML/src" revision="1.0.0" language="Java">',
        '')
    xml2 = xml2.replace(
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<unit xmlns="http://www.srcML.org/srcML/src" revision="1.0.0" language="Java">',
        '')
    # print(xml)
    print(get_sbt(xml2))