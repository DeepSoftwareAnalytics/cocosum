#!/usr/bin/env python
#!-*-coding:utf-8 -*- 
import javalang
import json
from tqdm import tqdm
import collections
import sys
sys.setrecursionlimit(2000)

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


def get_ast(code):
    ign_cnt = 0
    code = code.strip()
    tokens = javalang.tokenizer.tokenize(code)
    token_list = list(javalang.tokenizer.tokenize(code))
    length = len(token_list)
    parser = javalang.parser.Parser(tokens)
    try:
        tree = parser.parse_member_declaration()
    except (javalang.parser.JavaSyntaxError, IndexError, StopIteration, TypeError):
        # print(code)
        return None
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
            print(type(node))
            print(code)
            ign = True
            ign_cnt += 1
            # break
        outputs.append(d)
    return outputs

def SBT(r,ast_list):
    seq = []
    if not r.get("children"):
        seq.append("(")
        if r.get("value",False) :
            seq.append(r["type"]+"_"+ r.get("value"))
            seq.append(")")
            seq.append(r["type"]+"_"+ r.get("value"))
        else:
            seq.append(r["type"] )
            seq.append(")")
            seq.append(r["type"] )
    else:
        seq.append("(")
        if r.get("value",False) :
            seq.append(r["type"]+"_"+ r.get("value"))
        else:
            seq.append(r["type"])
        for node_index in r.get("children"):
            seq = seq + SBT(ast_list[node_index],ast_list)
        seq.append(")")
        if r.get("value",False):
            seq.append(r["type"] + "_" + r.get("value"))
        else:
            seq.append(r["type"])
    return seq

if __name__ == '__main__':
    # pre-process the source code: strings -> STR_, numbers-> NUM_, Booleans-> BOOL_
    # line = "public boolean doesNotHaveIds (){  return getIds () == null || getIds ().getIds().isEmpty(); }"
    line = "\tpublic int getPushesLowerbound() {\n\t\treturn pushesLowerbound;\n\t}\n"
    tokens = process_source(line)
    # generate ast file for source code
    code = " ".join(tokens)
    outputs = get_ast(code)
    print(outputs)
    root = outputs[0]
    sbt = SBT(root, outputs)
    print(sbt)
    str6 = " ".join(sbt)
    print(str6)