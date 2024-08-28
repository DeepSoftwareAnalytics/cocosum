# -*- coding: utf-8 -*-
# @Author : Lun
# @Created Time: Mon 24 Feb 2020 05:32:52 PM UTC
import pickle
import os
import argparse
import re

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
#ROOT_PATH = os.path.join(FILE_PATH, "..")
ROOT_PATH = FILE_PATH
DATA_PATH = os.path.join(ROOT_PATH, "data")
RES_PATH = os.path.join(ROOT_PATH, "res")
if os.path.exists(RES_PATH) == False:
    os.mkdir(RES_PATH)

def rfind_dot(s):
    p = -1
    for i in range(len(s) - 1, -1, -1):
        if s[i] == '.':
            p = i
            break
    return p
    
def main():
    """
    input:
    write: method2class.pkl, method2uml.pkl
    Example:
        python method2class.py --uml csn_mini/test/umls.pkl --method csn_mini/test/methods.pkl --output data/csn_mini/test --index csn_mini/test/method2uml_index.pkl
	    python method2class.py --uml csn_mini/train/umls.pkl --method csn_mini/train/methods.pkl --output data/csn_mini/train --index csn_mini/train/method2uml_index.pkl
	    python method2class.py --uml csn_mini/val/umls.pkl --method csn_mini/val/methods.pkl --output data/csn_mini/val--index csn_mini/val/method2uml_index.pkl
    """
    parser = argparse.ArgumentParser(formatter_class = argparse.RawTextHelpFormatter)
    parser.add_argument('--uml', type = str, required = True)
    parser.add_argument('--index', type = str, required = True)
    parser.add_argument('--method', type = str, required = True)
    
    parser.add_argument('--output', type = str)
    
    args = parser.parse_args()
    
    def load_pickle(path):
        path = os.path.join(DATA_PATH, path)
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data
        
    uml = load_pickle(args.uml)
    index = load_pickle(args.index)
    method = load_pickle(args.method)
    
    res = {}
    rindex = {}
    new_index = {}
    for key, value in index.items():
        if key not in method:
            continue
        new_index[key] = value
        if value in rindex:
            rindex[value].append(key)
        else:
            rindex[value] = [key]
    cnt = 0
    for key in rindex:
        rmapp = {re.sub(r"<.*>|\.", "", uml[key]["nodes_information"][i]["class_declaration"]["name"]): idx for idx, i in enumerate(uml[key]["nodes_information"].keys())}
        #print(key, rmapp)
        for m in rindex[key]:
            name = method[m]["func_name"]
            class_name = name[:rfind_dot(name)]
            if class_name not in rmapp:
                cnt += 1
                print(m, class_name, name, rmapp)
            else:
                res[m] = rmapp[class_name]
    print("[...] Saving")
    if args.output is None:
        outpath_classs = os.path.join(RES_PATH, "method2class.pkl")
        outpath_uml = os.path.join(RES_PATH, "method2uml.pkl")
    else:
        outpath_classs = os.path.join(args.output, "method2class.pkl")
        outpath_uml = os.path.join(args.output, "method2uml.pkl")
    with open(outpath_classs, "wb") as f:
        pickle.dump(res, f)
    with open(outpath_uml, "wb") as f:
        # print(new_index)
        # print(len(class_index), len(rindex))
        pickle.dump(new_index, f)
    print("[X] Saved.")
    print(cnt, len(index))

if __name__ == "__main__":
    main()
