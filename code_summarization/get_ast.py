# -*- coding: utf-8 -*-
# @Time      : 2020-03-24 12:35
# @Author    : Eason Hu
# @Site      : 
# @File      : get_ast.py

import javalang
import json

import collections
import sys

def process_source(line):
    code = line.strip()
    tokens = list(javalang.tokenizer.tokenize(code))
    tks = []
    for tk in tokens:
        if tk.__class__.__name__ == 'String' or tk.__class__.__name__ == 'Character':
            tks.append('STR_')
        elif 'Integer' in tk.__class__.__name__ or 'FloatingPoint' in tk.__class__.__name__:
            tks.append('NUM_')
        elif tk.__class__.__name__ == 'Boolean':
            tks.append('BOOL_')
        else:
            tks.append(tk.value)
    return tks

def getAst(line):
    code = line.strip()
    tokens = javalang.tokenizer.tokenize(code)
    token_list = list(javalang.tokenizer.tokenize(code))
    length = len(token_list)
    parser = javalang.parser.Parser(tokens)
    try:
        tree = parser.parse_member_declaration()
    except (javalang.parser.JavaSyntaxError, IndexError, StopIteration, TypeError):
        print(code)

    flatten = []
    for path, node in tree:
        flatten.append({'path': path, 'node': node})

    ign = False
    outputs = []
    stop = False
    for i, Node in enumerate(flatten):
        d = collections.OrderedDict()
        path = Node['path']
        node = Node['node']
        children = []
        for child in node.children:
            child_path = None
            if isinstance(child, javalang.ast.Node):
                child_path = path + tuple((node,))
                for j in range(i + 1, len(flatten)):
                    if child_path == flatten[j]['path'] and child == flatten[j]['node']:
                        children.append(j)
            if isinstance(child, list) and child:
                child_path = path + (node, child)
                for j in range(i + 1, len(flatten)):
                    if child_path == flatten[j]['path']:
                        children.append(j)
        d["id"] = i
        # d["type"] = str(node)
        d["type"] = str(node.__class__.__name__)
        if children:
            d["children"] = children
        value = None
        if hasattr(node, 'name'):
            value = node.name
        elif hasattr(node, 'value'):
            value = node.value
        elif hasattr(node, 'position') and node.position:
            for i, token in enumerate(token_list):
                if node.position == token.position:
                    pos = i + 1
                    value = str(token.value)
                    while (pos < length and token_list[pos].value == '.'):
                        value = value + '.' + token_list[pos + 1].value
                        pos += 2
                    break
        elif type(node) is javalang.tree.This \
                or type(node) is javalang.tree.ExplicitConstructorInvocation:
            value = 'this'
        elif type(node) is javalang.tree.BreakStatement:
            value = 'break'
        elif type(node) is javalang.tree.ContinueStatement:
            value = 'continue'
        elif type(node) is javalang.tree.TypeArgument:
            value = str(node.pattern_type)
        elif type(node) is javalang.tree.SuperMethodInvocation \
                or type(node) is javalang.tree.SuperMemberReference:
            value = 'super.' + str(node.member)
        elif type(node) is javalang.tree.Statement \
                or type(node) is javalang.tree.BlockStatement \
                or type(node) is javalang.tree.ForControl \
                or type(node) is javalang.tree.ArrayInitializer \
                or type(node) is javalang.tree.SwitchStatementCase:
            value = 'None'
        elif type(node) is javalang.tree.VoidClassReference:
            value = 'void.class'
        elif type(node) is javalang.tree.SuperConstructorInvocation:
            value = 'super'

        if value is not None and type(value) is type('str'):
            d['value'] = value
        if not children and not value:
            # print('Leaf has no value!')
            print(type(node))
            print(code)
            ign = True
        outputs.append(d)
    if not ign:
        return outputs


def build_ast(code):
    # code_list = process_source(code)
    output = getAst(code)
    return output

def traverse_java_tree2(tree):
    new_tree = {}

    for d in tree:
        if d.get('children', False):
            cur_dict = {}
            id_name = '1*NODEFIX' + str(d['id'])
            cur_dict['node'] = d['value'] if d.get('value', False) else d['type']
            cur_dict['children'] = ['1*NODEFIX' + str(idx) for idx in d['children']]
            new_tree[id_name] = cur_dict
        else:
            cur_dict = {}
            id_name = '1*NODEFIX' + str(d['id'])
            cur_dict['node'] = d['value'] if d.get('value', False) else d['type']
            new_tree[id_name] = cur_dict


    for cur_id in new_tree:
        if new_tree[cur_id].get('children', False):
            for child_id in new_tree[cur_id]['children']:
                new_tree[child_id]['parent'] = cur_id


    for cur_id in new_tree:
        if not new_tree[cur_id].get('children', False):
            new_tree[cur_id]['children'] = [new_tree[cur_id]['node']]
            # new_tree[new_tree[cur_id]['parent']]['children'].remove(cur_id)
            # new_tree[new_tree[cur_id]['parent']]['children'].append(new_tree[cur_id]['node'])
        if not new_tree[cur_id].get('parent', False):
            new_tree[cur_id]['parent'] = None
    # return_tree = {}
    # for cur_id in new_tree:
    #     if new_tree[cur_id].get('children', False):
    #         return_tree[cur_id] = new_tree[cur_id]
    return new_tree