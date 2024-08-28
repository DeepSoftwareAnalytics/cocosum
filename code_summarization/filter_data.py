import torch
from lib.data.Tree import *
data_path = './csn_dataset/train_data/processed_fid_tgt_10.train.pt' # this file have fid
data = torch.load(data_path)
print('finished data load')

def get_data_trees(trees):
    data_trees = []
    for t_json in trees:
        # for k, node in t_json.iteritems():
        #     # if 'parent' not in node:
        #     #     root_idx ='#'
        #     #     break
        #     if node['parent'] == None and node['node'] != 'StatementExpression':
        #         root_idx = k
        # if root_idx == "#":
        #     continue
        tree = json2tree_binary(t_json, Tree(), '1*NODEFIX0')
        data_trees.append(tree)

    return data_trees

def get_data_trees_from_dict(data):
    data_trees = {}
    for fid in data:
        t_json = data[fid]['trees']
        # for k, node in t_json.iteritems():
        #     # if 'parent' not in node:
        #     #     root_idx ='#'
        #     #     break
        #     if node['parent'] == None and node['node'] != 'StatementExpression':
        #         root_idx = k
        # if root_idx == "#":
        #     continue
        tree = json2tree_binary(t_json, Tree(), '1*NODEFIX0')
        data_trees[fid] = tree

    return data_trees
new_data = {'train_xe': {}, 'test': {}, 'train_pg': {}, 'valid': {}}

key = ['train_xe', 'test', 'valid']
for k in key:
    print('processing original', k)
    print('original data length', len(data[k]))
    src = []
    tgt = []
    trees = []
    fids_list = []
    binary_tree = get_data_trees_from_dict(data[k])
    for fid in data[k]:
        if binary_tree[fid].leaf_count() > 1:
            fids_list.append(fid)
            src.append(data[k][fid]['src'])
            tgt.append(data[k][fid]['tgt'])
            trees.append(data[k][fid]['trees'])

    new_data[k]['fid'] = fids_list
    new_data[k]['src'] = src
    new_data[k]['tgt'] = tgt
    new_data[k]['trees'] = trees
print('finished process original data')
# print('###############')
# print('verify data is right or not')

error = 0
for k in key:
    print('verify data', k)
    verify_tree = get_data_trees(new_data[k]['trees'])
    # print('new data length', len(data[k]), k)
    print('tree length ', len(verify_tree))
    print('data length', len(new_data[k]['tgt']), len(new_data[k]['trees']), len(new_data[k]['src']))
    # for fid in data[k]:
    #     if verify_tree[fid].leaf_count <= 1:
    #         print('---------------')
    #         print(fid)
    #         error += 1
    i = 0
    while i < len(new_data[k]['trees']):
        if verify_tree[i].leaf_count() > 1:
            i += 1
        else:
            print('--------------------')
            print(i)
            error += 1
            i += 1

print('error', error)
assert error == 0


new_data['dicts'] = data['dicts']
new_data['train_pg'] = new_data['train_xe']
torch.save(new_data, './csn_dataset/train_data/processed_fid_filter_tgt_10.train.pt') # filter which leaf_count <= 1
