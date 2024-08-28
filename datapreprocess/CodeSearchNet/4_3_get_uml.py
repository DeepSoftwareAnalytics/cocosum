# !/usr/bin/env python
# !-*-coding:utf-8 -*-
import os
import networkx as nx
from networkx.drawing.nx_pydot import read_dot
from bs4 import BeautifulSoup
import time
import traceback
import pickle
import re
from multiprocessing import Pool, cpu_count
import sys
sys.path.append("../../")
from util.Config import Config as cf
from util.LoggerUtil import set_logger, debug_logger
from util.DataUtil import get_file_size, save_pickle_data, time_format, make_directory
import natsort


def get_class_declaration(html_str):
    """
    parse class declaration
    class_declaration: dict {name: xxx type: xxx}
    """
    if len(html_str) > 3:
        label = html_str[-3]
    else:
        label = html_str[1]
    class_declaration = {"name": "", "type": Config.class_type[0]}
    for tr in label .findAll('tr'):
        for td in tr.findAll('td'):
            if "arial italic" in str(td):
                class_declaration["type"] = Config.class_type[1]
            elif "«interface»" in td.getText():
                class_declaration["type"] = Config.class_type[2]
            class_declaration["name"] = td.getText().strip()
    return class_declaration


def get_class_attributes(html_str):
    """
    parser attribute
    attributes: list[dict0, dict1, ]
    dict0: {name:xxx, access:xxx, variable_type: xxx}
    """
    attributes = []
    for tr in html_str[-2].findAll('tr'):
        attribute = {}
        for td in tr.findAll('td'):
            text_token = td.getText().split()
            if text_token:
                attribute["access"] = Config.access_type.get(text_token[0], "others")
                attribute["name"] = text_token[1].strip()
                attribute["type"] = text_token[-1]
        attributes.append(attribute)
    return attributes


def get_method_signature(txt, access):
    signature = txt.strip()
    signature = access + signature[1:]
    return signature


def get_method_name(txt):
    rm_parameter = re.sub(u"\\(.*?\\)|\\{.*?}|\\[.*?]", "", txt)  # Remove the parameters list
    rm_parameter_token = rm_parameter.split()
    # method["name"] = rm_parameter_token[1]
    return rm_parameter_token[1], rm_parameter_token


def format_method_information(html_tr):
    method = {}
    for td in html_tr.findAll('td'):
        txt = td.getText()
        method_inforamation = txt.split()

        if len(method_inforamation) > 1:
            if "arial italic" in str(td):
                method["type"] = Config.class_type[1]

            method["access"] = Config.access_type.get(method_inforamation[0], "others")
            method["signature"] = get_method_signature(txt, method["access"])
            pattern = re.compile(r'[(](.*)[)]', re.S)  # Match bracket to get parameters list
            method["Parameters"] = re.findall(pattern, txt)
            method["name"], parameter_token = get_method_name(txt)
            if len(parameter_token) == 4:
                method["return_type"] = parameter_token[-1]
            elif len(parameter_token) == 2:
                method["return_type"] = "void"
    return method


# Parser node in the specific form
def get_class_information(html_tables):
    """
    parser each node into tree parts: class_declaration, attributes:  methods:
    class_declaration : dict{ name: xxxtype: xxx}
    attributes: list[dict0,dict1,] dict0:{ name:xxx, access:xxx, variable_type: xxx}
    methods: list[dict0,dict1,]
            dict0:{ type:xxx, signature:xxx, name:xxx, access:xxx, return_type: xxx ,parameters: str}
    """
    # dict {class_declaration: xxx , attributes: xxx,   methods: xxx }
    class_information = {"class_declaration": get_class_declaration(html_tables)}
    methods = []
    if len(html_tables) > 3:
        class_information["attributes"] = get_class_attributes(html_tables)
        methods = [format_method_information(tr) for tr in html_tables[-1].findAll('tr')]
    class_information["methods"] = methods

    return class_information


# Get the  relation type between two classed
def get_edge_type(edge):
    relationship_type = ["DEPEND", "NAVASSOC", "ASSOC", "NAVHAS", "HAS",
                         "NAVCOMPOSED", "COMPOSED", "IMPLEMENTS", "EXTENDS"]
    if edge.get("arrowhead", "wu") == "open":
        if edge.get("style", "wu") == "dashed":
            edge["realtiontype"] = relationship_type[0]
        else:
            if edge.get("arrowtail", "wu") == "diamond":
                edge["realtiontype"] = relationship_type[5]
            elif edge.get("arrowtail", "wu") == "ediamon":
                edge["realtiontype"] = relationship_type[3]
            else:
                edge["realtiontype"] = relationship_type[1]
    elif edge.get("arrowhead", "wu") == "none":
        if edge.get("arrowtail", "wu") == "diamond":
            edge["realtiontype"] = relationship_type[6]
        elif edge.get("arrowtail", "wu") == "ediamon":
            edge["realtiontype"] = relationship_type[4]
        else:
            edge["realtiontype"] = relationship_type[2]
    else:
        if edge.get("style", "wu") == "dashed":
            edge["realtiontype"] = relationship_type[7]
        else:
            edge["realtiontype"] = relationship_type[8]
    return edge["realtiontype"]


# Process the nodes to get the non-duplicated nodes
def merge_reduplicated_nodes(graph):
    merge_nodes = {}
    new_nodes = []

    nodes_information = dict(graph.nodes(True))
    old_nodes = list(graph.nodes)  # ['c0', 'c1','c0:p', 'c1:p']

    for node in old_nodes:
        if not nodes_information[node]:
            pos = node.find(":")
            new_node = node[:pos]
            merge_nodes[node] = new_node
        else:
            new_nodes.append(node)

    return new_nodes, merge_nodes


def format_edge(edges, nodes):
    return [(nodes[edge[0]], nodes[edge[1]]) for edge in edges]


def format_node_information(ori_nodes_label,nodes):
    nodes_label = {}
    for node in nodes:
        text = ori_nodes_label[node]["label"]
        # https://beautiful-soup-4.readthedocs.io/en/latest/
        soup = BeautifulSoup(text, features="lxml")
        tables = soup.findAll('table')
        nodes_label[node] = get_class_information(tables)
    return nodes_label


def format_edge_information(edges_label, nodes):
    # edges_information = list(graph.edges(data=True))
   return [[nodes[edge[0]], nodes[edge[1]],
            {"relationtype": get_edge_type(edge[2])}] for edge in edges_label]


#  Given a dot file, return the uml dict
def format_uml(dot_file):
    """
    Given a dot file, return the uml dict

    keys of uml ['nodes', 'number_of_nodes', 'reduplicate_nodes_dict', 'edges',
    'number_of_edges', 'nodes_information', 'edge_information']

    uml {'nodes':['c0', 'c1'], 'number_of_nodes':2,
    'reduplicate_nodes_dict':{'c0:p': 'c0', 'c1:p': 'c1'},
    'edges': [('c0', 'c1')], 'number_of_edges':1,
    'nodes_information:xxx',
    'edge_information:edge_information': [[' Person ', ' House ', {'relation type': 'NAVASSOC'}],'}
    """
    uml = {}
    # https://networkx.github.io/documentation/stable/tutorial.html#multigraphs
    graph = nx.MultiGraph(read_dot(dot_file))
    uml["nodes"], uml["reduplicated_nodes"] = merge_reduplicated_nodes(graph)
    uml["number_of_nodes"] = len(uml["nodes"])

    uml["edges"] = format_edge(list(graph.edges), uml["reduplicated_nodes"])
    uml["number_of_edges"] = len(uml["edges"])

    uml["nodes_information"] = format_node_information(dict(graph.nodes(True)), uml["nodes"])
    uml["edge_information"] = format_edge_information(list(graph.edges(data=True)),  uml["reduplicated_nodes"])

    return uml


# Replace the "#" with "$"
def rewrite_dot_file(dot_file):
    dot_name = dot_file.split("/")[-1]
    tem_dot_file = os.path.join(Config.rewrite_dot_path, Config.part, dot_name + "_rewrite" + ".dot")
    dot_str = ""

    with open(dot_file) as f:
        flag = True
        for line in f.readlines():
            if flag:
                if line.startswith("#"):
                    continue
                else:
                    flag = False
                    continue
            line = line.strip().replace("#", "$")
            dot_str += line + "\r"

    with open(tem_dot_file, "w") as output:
        output.write(dot_str)

    return tem_dot_file


def process_dot_file(file_name):
    try:
        dot_file_path = os.path.join(Config.dot_path, Config.part, file_name)
        # TODO: why big_graph? continue. # what is 500?
        if get_file_size(dot_file_path) > Config.big_graph_node_cnt:
            return "BigG"
        new_dot_file = rewrite_dot_file(dot_file_path)
        return format_uml(new_dot_file)
    except:
        error_file = os.path.join(Config.uml_dir, part + "_error.txt")
        traceback.print_exc(file=open(error_file, 'a'))
        return "Error"


def get_uml_data():

    dot_path = os.path.join(Config.dot_path, Config.part)
    dot_files = list(natsort.natsorted(os.listdir(dot_path)))
    methods, umls, big_graph_id = {}, {}, []

    p = Pool(cpu_count())
    results = p.map(process_dot_file, dot_files)
    p.close()
    p.join()

    for k, v in zip(dot_files, results):
        fid = int(k.split(".")[0])
        if v == "BigG":
            big_graph_id.append(fid)
        elif type(v) == dict:
            umls[fid] = v
        # else:
        #     print("error")

    uml_dir = os.path.join(Config.uml_dir, Config.part)
    save_pickle_data(os.path.join(uml_dir), 'umls.pkl', umls)
    save_pickle_data(os.path.join(uml_dir), 'big_graph_id.pkl', big_graph_id)


class Config(object):
    data_path = "../../data/csn/csn.pkl"
    dot_path = "../../data/csn/uml/dot/"
    rewrite_dot_path = "../../data/csn/uml/rew_dot/"
    uml_dir = "../../data/csn/uml/"
    big_graph_node_cnt = 500
    part = "train"
    access_type = {'+': 'public', '-': 'private', '$': 'protected'}
    class_type = ["concrete", "abstract", "interface"]


if __name__ == '__main__':
    set_logger(cf.DEBUG)
    start_time = time.perf_counter()
    parts = ['train', 'test', 'valid']
    # parts = ['valid']
    for part in parts:
        Config.part = part
        make_directory(os.path.join(Config.rewrite_dot_path,part))
        get_uml_data()
    debug_logger("Time cost %s" % time_format(time.perf_counter() - start_time))
