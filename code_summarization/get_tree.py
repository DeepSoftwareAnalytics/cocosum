# -*- coding: utf-8 -*-
# @Time      : 2020-03-24 12:25
# @Author    : Eason Hu
# @Site      : 
# @File      : gettree.py

# -*- coding: utf-8 -*-
# @Time      : 2020-03-24 12:11
# @Author    : Eason Hu
# @Site      :
# @File      : test_getTree.py

import lib
import argparse
import torch
import codecs
import lib.data.Constants as Constants
import ast, asttokens
import sys
from lib.data.Tree import *
from get_ast import *
import re
import gensim
# from .Dict import Dict

lang = 'javalang' # java, javalang
trees = []

def get_data_trees(trees):
    data_trees = []
    for t_json in trees:
        for k, node in t_json.iteritems():
            if node['parent'] == None:
                root_idx = k
        tree = json2tree_binary(t_json, Tree(), root_idx)
        data_trees.append(tree)

    return data_trees

if lang == 'java':
    sline = 'protected final void bindIndexed(ConfigurationPropertyName name, Bindable<?> target,\n\t\t\tAggregateElementBinder elementBinder, ResolvableType aggregateType,\n\t\t\tResolvableType elementType, IndexedCollectionSupplier result) {\n\t\tfor (ConfigurationPropertySource source : getContext().getSources()) {\n\t\t\tbindIndexed(source, name, target, elementBinder, result, aggregateType,\n\t\t\t\t\telementType);\n\t\t\tif (result.wasSupplied() && result.get() != null) {\n\t\t\t\treturn;\n\t\t\t}\n\t\t}\n\t}'
    tree = java2tree(sline)
    tree_json = {}
    tree_json, _ = traverse_java_tree(tree, tree_json)

if lang == 'python':
    sline = "def get_flashed_messages(with_categories=False, category_filter=[]): DCNL  DCSP flashes = _request_ctx_stack.top.flashes DCNL DCSP if (flashes is None): DCNL DCSP  DCSP _request_ctx_stack.top.flashes = flashes = (session.pop('_flashes') if ('_flashes' in session) else []) DCNL DCSP if category_filter: DCNL DCSP  DCSP flashes = list(filter((lambda f: (f[0] in category_filter)), flashes)) DCNL DCSP if (not with_categories): DCNL DCSP  DCSP return [x[1] for x in flashes] DCNL DCSP return flashes"
    # srcLine = python_tokenize(sline.replace(' DCNL DCSP ', '').replace(' DCNL ', '').replace(' DCSP ', ''))
    sline = sline.replace(' DCNL DCSP ', '\n\t').replace(' DCNL  DCSP ', '\n\t').replace(' DCNL   DCSP ','\n\t').replace(' DCNL ','\n').replace(' DCSP ', '\t')

    atok, tree = python2tree(sline)
    tree_json = traverse_python_tree(atok, tree)

if lang == 'javalang':
    sline = 'protected final void bindIndexed(ConfigurationPropertyName name, Bindable<?> target,\n\t\t\tAggregateElementBinder elementBinder, ResolvableType aggregateType,\n\t\t\tResolvableType elementType, IndexedCollectionSupplier result) {\n\t\tfor (ConfigurationPropertySource source : getContext().getSources()) {\n\t\t\tbindIndexed(source, name, target, elementBinder, result, aggregateType,\n\t\t\t\t\telementType);\n\t\t\tif (result.wasSupplied() && result.get() != null) {\n\t\t\t\treturn;\n\t\t\t}\n\t\t}\n\t}'
    tree = ast(sline)
    tree_json = traverse_java_tree2(tree)
tree_json = split_tree(tree_json, len(tree_json))
tree_json = merge_tree(tree_json)
# if len(tree_json) < opt.src_seq_length:
trees += [tree_json]

print(get_data_trees(trees))
