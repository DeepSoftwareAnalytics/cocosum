import torch
from lib.data.Tree import *


data_path = '/mnt/yuxuan/Code_summarization/csn_dataset/train_data/processed_filter.train.pt2' # this file have been filtered
data = torch.load(data_path)
print('finished data load')


def get_data_trees(trees):
    data_trees = []
    for t_json in trees:
        # for k, node in t_json.iteritems():
        #     if node['parent'] == None:
        #         root_idx = k
        # if root_idx == "#":
        #     continue
        tree = json2tree_binary(t_json, Tree(), '1*NODEFIX0')
        data_trees.append(tree)

    return data_trees


error = 0
key = ['train_xe', 'test', 'valid']
for k in key:
    print('verify data', k)
    verify_tree = get_data_trees(data[k]['trees'])
    # print('new data length', len(data[k]), k)
    print('tree length ', len(verify_tree))
    print('data length', len(data[k]['tgt']), len(data[k]['trees']), len(data[k]['src']))
    # for fid in data[k]:
    #     if verify_tree[fid].leaf_count <= 1:
    #         print('---------------')
    #         print(fid)
    #         error += 1
    i = 0
    while i < len(data[k]['trees']):
        if verify_tree[i].leaf_count() > 1:
            i += 1
        else:
            print('--------------------')
            print(i)
            error += 1
            i += 1

print('error', error)
assert error == 0