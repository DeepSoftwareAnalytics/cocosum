# -*- coding: utf-8 -*-
# @Author : Lun
# @Created Time: Mon 24 Feb 2020 05:32:52 PM UTC
import pickle
import os
import argparse
import re
import sys

import torch
sys.path.append("../../")
from util.DataUtil import read_pickle_data, time_format
# FILE_PATH = os.path.dirname(os.path.abspath(__file__))
# # ROOT_PATH = os.path.join(FILE_PATH, "..")
# ROOT_PATH = FILE_PATH
# DATA_PATH = os.path.join(ROOT_PATH, "data")
# RES_PATH = os.path.join(ROOT_PATH, "res")
# if os.path.exists(RES_PATH) == False:
#     os.mkdir(RES_PATH)


# def rfind_dot(s):
#     p = -1
#     for i in range(len(s) - 1, -1, -1):
#         if s[i] == '.':
#             p = i
#             break
#     return p


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    # parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--uml_home', type=str, required=True)
    parser.add_argument('--output', type=str)

    args = parser.parse_args()

    # def load_pickle(path):
    #     path = os.path.join(DATA_PATH, path)
    #     with open(path, "rb") as f:
    #         data = pickle.load(f)
    #     return data

    print("[...] Loading")

    train_m2u = read_pickle_data(os.path.join(args.uml_home, "train/method2uml.pkl"))
    test_m2u = read_pickle_data(os.path.join(args.uml_home, "test/method2uml.pkl"))
    val_m2u = read_pickle_data(os.path.join(args.uml_home, "valid/method2uml.pkl"))

    train_m2c = read_pickle_data(os.path.join(args.uml_home, "train/method2class.pkl"))
    test_m2c = read_pickle_data(os.path.join(args.uml_home, "test/method2class.pkl"))
    val_m2c = read_pickle_data(os.path.join(args.uml_home, "valid/method2class.pkl"))

    uml_dataset = {"train": torch.load(os.path.join(args.uml_home, "train/uml_dataset.pt")),
                   "valid": torch.load(os.path.join(args.uml_home, "valid/uml_dataset.pt")),
                   "test": torch.load(os.path.join(args.uml_home, "test/uml_dataset.pt"))}

    m2u_m2c = {"m2utrain": train_m2u, "m2uval": val_m2u, "m2utest": test_m2u, "m2ctrain": train_m2c, "m2cval": val_m2c,
               "m2ctest": test_m2c}

    print("[X] Loaded.")

    print("[...] Saving")
    if args.output is None:
        output_m2u_m2c_path = os.path.join(args.uml_home, "m2u_m2c.pkl")
        output_uml_pt_path = os.path.join(args.uml_home, "uml_dataset.pt")
    else:
        output_m2u_m2c_path = os.path.join(args.output, "m2u_m2c.pkl")
        output_uml_pt_path = os.path.join(args.output, "uml_dataset.pt")

    with open(output_m2u_m2c_path, "wb") as f:
        pickle.dump(m2u_m2c, f)
    torch.save(uml_dataset, output_uml_pt_path)
    print("[X] Saved.")


if __name__ == "__main__":
    main()
