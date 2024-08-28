# -*- coding: utf-8 -*-
# @Author : Lun
# @Created Time: Mon 24 Feb 2020 05:32:52 PM UTC
import pickle
import os
import argparse
import re

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
# ROOT_PATH = os.path.join(FILE_PATH, "..")
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
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--uml_home', type=str, required=True)

    parser.add_argument('--output', type=str)

    args = parser.parse_args()

    def load_pickle(path):
        path = os.path.join(DATA_PATH, path)
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data

    print("[...] Loading")

    dataset = load_pickle(args.dataset)
    train_m2u = load_pickle(os.path.join(args.uml_home, "train/method2uml.pkl"))
    test_m2u = load_pickle(os.path.join(args.uml_home, "test/method2uml.pkl"))
    # val_m2u = load_pickle(os.path.join(args.uml_home, "val/method2uml.pkl"))
    val_m2u = load_pickle(os.path.join(args.uml_home, "valid/method2uml.pkl"))
    train_m2c = load_pickle(os.path.join(args.uml_home, "train/method2class.pkl"))
    test_m2c = load_pickle(os.path.join(args.uml_home, "test/method2class.pkl"))
    # val_m2c = load_pickle(os.path.join(args.uml_home, "val/method2class.pkl"))
    val_m2c = load_pickle(os.path.join(args.uml_home, "valid/method2class.pkl"))

    dataset["m2utrain"] = train_m2u
    dataset["m2uval"] = val_m2u
    dataset["m2utest"] = test_m2u

    dataset["m2ctrain"] = train_m2c
    dataset["m2cval"] = val_m2c
    dataset["m2ctest"] = test_m2c
    print("[X] Loaded.")

    print("[...] Saving")
    if args.output is None:
        outpath = os.path.join(RES_PATH, "dataset_uml.pkl")
    else:
        outpath = os.path.join(args.output, "dataset_uml.pkl")
    with open(outpath, "wb") as f:
        pickle.dump(dataset, f)
    print("[X] Saved.")


if __name__ == "__main__":
    main()
