import pickle
import json
import os
import sys
import argparse


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters  
    parser.add_argument("--filename", default="../csn_mini_data/dlen100_clen12_slen435_dvoc10000_cvoc10000_svoc10000_dataset.pkl", type=str, required=False,
                        help="input pickle file name" )   
    parser.add_argument("--outdir", default='json_output', type=str, required=False,
                        help="The output directory where the json/jsonl files will be written.")
    
    args = parser.parse_args()
    filename = args.filename
    outdir = args.outdir
    os.makedirs(outdir, exist_ok = True)
    dataset = pickle.load(open(filename, "rb"))
    print(dataset.keys())

    # comment, data, sbt to jsonl
    basename = filename.split('.')[0]
    for tag in ['train', 'val', 'test']:
        outfile = "{}/{}.{}.jsonl".format(outdir, 'data', tag)
        with open(outfile, 'w') as f:
            for k in dataset['c'+tag]:
                sample = {'fid': k,
                                   'c': dataset['c'+tag][k],
                                   'd': dataset['d'+tag][k],
                                   's': dataset['s'+tag][k]}
                f.write(json.dumps(sample)+'\n')


    # comstok, datstok, smlstok to json
    for tag in ['comstok', 'datstok', 'smlstok']:
        with open("{}/{}.json".format(outdir, tag), 'w') as f:
            f.write(json.dumps(dataset[tag]))


    # m2u, m2c to jsonl
    outdir = 'json_output'
    for tag in ['train', 'val', 'test']:
        outfile = "{}/{}.{}.jsonl".format(outdir, 'm2u_m2c', tag)
        with open(outfile, 'w') as f:
            for k in dataset['m2u'+tag]:
                sample = {'fid': k,
                                   'm2u': dataset['m2u'+tag][k],
                                   'm2c': dataset['m2c'+tag][k]}
                f.write(json.dumps(sample)+'\n')
                
                
                
                
if __name__ == "__main__":
    main() 