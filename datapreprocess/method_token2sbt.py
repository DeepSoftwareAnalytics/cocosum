#!/usr/bin/env python
# !-*-coding:utf-8 -*-

"""
This script will read the json file of code search net
, obtain sbt of method token  and add the sbt to data set.

If you run this script , please set three parameters:
old_data_path, new_data_name, new_data_path in the function config();

"""

import json
import os
import time
import javalang
import collections
import sys
from multiprocessing import Pool
import traceback
# Set the maximum depth of recursively processing, which traverse ast to obtain sbt
sys.setrecursionlimit(2000)


# Open file with file path
def open_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return lines


# Copy from https://github.com/xing-hu/EMSE-DeepCom/blob/master/data_utils/get_ast.py#L28-L122
def get_ast(code):
    ign_cnt = 0
    code = code.strip()
    tokens = javalang.tokenizer.tokenize(code)
    token_list = list(javalang.tokenizer.tokenize(code))
    length = len(token_list)
    parser = javalang.parser.Parser(tokens)
    try:
        tree = parser.parse_member_declaration()
        # tree = parser.parse_method_or_field_declaraction()
    except (javalang.parser.JavaSyntaxError, IndexError, StopIteration, TypeError):
        # print(code)
        pass
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
            # print(type(node))
            # print(code)
            ign = True
            ign_cnt += 1
            # break
        outputs.append(d)
    return outputs


# Implement the algorithm 1 ( Structure-based Traversal )
# https://xin-xia.github.io/publication/icpc182.pdf
# The parameter root means the root of sub-tree
def get_sbt(root_of_sub_tree, ast):

    sbt = []

    if not root_of_sub_tree.get("children"):

        sbt.append("(")

        if root_of_sub_tree.get("value", False):
            sbt.append(root_of_sub_tree["type"] + "_" + root_of_sub_tree.get("value"))
            sbt.append(")")
            sbt.append(root_of_sub_tree["type"] + "_" + root_of_sub_tree.get("value"))
        else:
            sbt.append(root_of_sub_tree["type"])
            sbt.append(")")
            sbt.append(root_of_sub_tree["type"])

    else:

        sbt.append("(")

        if root_of_sub_tree.get("value", False):
            sbt.append(root_of_sub_tree["type"] + "_" + root_of_sub_tree.get("value"))
        else:
            sbt.append(root_of_sub_tree["type"])

        # Recursive call get_sbt
        for node_index in root_of_sub_tree.get("children"):
            sbt = sbt + get_sbt(ast[node_index], ast)
        sbt.append(")")

        if root_of_sub_tree.get("value", False):
            sbt.append(root_of_sub_tree["type"] + "_" + root_of_sub_tree.get("value"))
        else:
            sbt.append(root_of_sub_tree["type"])

    return sbt


#    write list object to json file
def write_list_to_json(data, file_name, file_save_path):

    os.chdir(file_save_path)
    with open(file_name, 'w') as f:
        json.dump(data, f)


# Old_data could be found in https://github.com/github/CodeSearchNet/blob/master/notebooks/ExploreData.ipynb
# The main() function loads the old file and obtains sbt from method token of each old data
def main_function(old_data_path, new_data_name, new_data_path):
    old_data = open_file(old_data_path)
    new_data = []
    # It is not the same as the length of old_data, because some old data can't extract uml information
    available_data_number = 0

    # Obtain sbt of method token of each data and add to data
    for data in old_data:
        data = json.loads(data)
        # Obtain method code snippet by joining the code_tokens using white space
        code_tokens = data["code_tokens"]
        code = " ".join(code_tokens)

        available_data_number = available_data_number + 1
        try:
            # Obtain ast  of the code
            ast = get_ast(code)
            root = ast[0]
            data["SBT"] = get_sbt(root, ast)
            new_data.append(data)
            # print(new_data_name," available_data_number  ",    available_data_number )
        except:
            error_file = "exact_sbt_error.txt"
            traceback.print_exc(file=open(error_file, 'a'))
            continue

    print(new_data_name, len(new_data))
    write_list_to_json(new_data, new_data_name, new_data_path)


# make directory for given path
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    else:
        pass


class Config(object):

    new_data_name = ["java_train_" + str(i) + ".jsonl" for i in range(16)]
    new_data_name.extend(["java_valid_0.jsonl", "java_test_0.jsonl"])

    old_data_path = [r"/datadrive/MyCodeSumNet/MethodSum/data/java/java/final/jsonl/train/java_train_" + str(i) + ".jsonl"
                     for i in range(16)]
    old_data_path.append(r"/datadrive/MyCodeSumNet/MethodSum/data/java/java/final/jsonl/valid/java_valid_0.jsonl")
    old_data_path.append(r"/datadrive/MyCodeSumNet/MethodSum/data/java/java/final/jsonl/test/java_test_0.jsonl")

    new_data_path = [r"/datadrive/MyCodeSumNet/MethodSum/data/java/java/final/SBT_jsonl/train/" for i in range(16)]
    new_data_path.append(r"/datadrive/MyCodeSumNet/MethodSum/data/java/java/final/SBT_jsonl/valid/")
    new_data_path.append(r"/datadrive/MyCodeSumNet/MethodSum/data/java/java/final/SBT_jsonl/test/")


if __name__ == '__main__':

    debug = True
    start_time = time.perf_counter()

    if debug is True:
        old_data_file_path = r"D:\MSRA\data\java\final\jsonl\train\java_train.jsonl"
        new_data_file_name = "java_train.jsonl"
        new_data_file_path = r"D:\MSRA\data\java\final\jsonl\train\final\sbt_jsonl\train"
        isExists = os.path.exists(new_data_file_path)
        if not isExists:
            mkdir(new_data_file_path)
        main_function(old_data_file_path, new_data_file_name, new_data_file_path)

    else:
        # Apply multiprocessing to process each data set
        # Difference with the above two lines's code, the following code runs in ubuntu.
        p = Pool(18)

        for i in range(len(Config.old_data_path)):
            isExists = os.path.exists(Config.new_data_path[i])
            if not isExists:
                mkdir(Config.new_data_path[i])
            p.apply_async(main_function, args=(Config.old_data_path[i], Config.new_data_name[i], Config.new_data_path[i]))

        p.close()  # Unable to continue adding new processes after p.close()
        p.join()  # It guarantee that the main process does not end until the whole sub-process is finished

    end_time = time.perf_counter()
    # Convert data in seconds to a hour:minute:second format
    seconds = end_time - start_time
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    print(seconds)
    print("%02d:%02d:%02d" % (h, m, s))
