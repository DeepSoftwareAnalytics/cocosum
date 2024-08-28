# !/usr/bin/env python
# !-*-coding:utf-8 -*-
import os
import networkx as nx
from networkx.drawing.nx_pydot import read_dot
from bs4 import BeautifulSoup
import subprocess
import time
import traceback
import pickle
import re
from multiprocessing import Pool, cpu_count
import sys
sys.path.append("../../")
from util.DataUtil import make_directory, get_file_size, save_pickle_data


# Open a file to obtain the package name
def get_package_name(path_of_java_file):
    package = ""
    with open(path_of_java_file) as file:
        lines = file.readlines()

    # Find the package name
    for line in lines:
        if "package" in line[:7]:
            package = line.replace("package", "")
            package = package.replace(";\n", "")
            package = package.replace(";", "")
            package = package.replace(" ", "")
            break

    return package


# Get package absolute path
def get_package_path(java_fun_path, package_name):
    package_relative_path = package_name.replace('.', "/")
    pos = java_fun_path.find(package_relative_path)
    package_path = java_fun_path[:pos]

    return package_path


# Parser node in the specific form
def parser_node(tables):
    """
    parser each node into tree parts: class_declaration, attributes:  methods:
    class_declaration : dict{ name: xxx​ type: xxx}​
    attributes: list​​[dict0,dict1,] dict0:{ name:​xxx, access:​xxx, variable_type: xxx​​}
    methods: list​[dict0,dict1,]
            dict0:{ type:​xxx, signature:​xxx, name:​xxx, access:​xxx, return_type: xxx ,parameters: str}
    """
    access_type = {'+': 'public', '-': 'private', '$': 'protected'}
    class_type = ["concrete", "abstract", "interface"]

    # obtain the information of class in detail
    class_information = {}  # dict {class_declaration: xxx , attributes: xxx,   methods: xxx }
    class_declaration, attributes, methods = {}, [], []

    # parse class declaration
    class_declaration["name"] = ""
    class_declaration["type"] = class_type[0]

    if len(tables) > 3:
        # parse class declaration
        # class_declaration: dict {name: xxx​ type: xxx}​
        for tr in tables[-3].findAll('tr'):
            for td in tr.findAll('td'):
                if "arial italic" in str(td):
                    class_declaration["type"] = class_type[1]
                elif "«interface»" in td.getText():
                    class_declaration["type"] = class_type[2]
                class_declaration["name"] = td.getText().strip()
        class_information["class_declaration"] = class_declaration

        # parser attribute
        # attributes: list​​[dict0, dict1, ]
        # dict0: {name:​xxx, access:​xxx, variable_type: xxx​​}
        for tr in tables[-2].findAll('tr'):
            attribute = {}
            for td in tr.findAll('td'):
                text_token = td.getText().split()
                if text_token:
                    attribute["access"] = access_type.get(text_token[0], "others")
                    attribute["name"] = text_token[1].strip()
                    attribute["type"] = text_token[-1]
            attributes.append(attribute)
        class_information["attributes"] = attributes

        # parser method
        # methods: list​[dict0, dict1, ]
        # dict0: {type:​xxx, signature:​xxx, name:​xxx, access:​xxx, return_type: xxx, parameters: str}

        for tr in tables[-1].findAll('tr'):
            method = {}
            for td in tr.findAll('td'):
                txt = td.getText()
                method_inforamation = txt.split()

                if len(method_inforamation) > 1:
                    # Obtain type  of method
                    if "arial italic" in str(td):
                        method["type"] = class_type[1]

                    # Obtain access of method
                    method["access"] = access_type.get(method_inforamation[0], "others")

                    # Obtain signature of method
                    signature = txt.strip()
                    signature = method["access"] + signature[1:]
                    method["signature"] = signature

                    # Obtain Parameters of method
                    pattern = re.compile(r'[(](.*)[)]', re.S)  # Match bracket to get parameters list
                    method["Parameters"] = re.findall(pattern, txt)

                    # Obtain Parameters of method
                    rm_parameter = re.sub(u"\\(.*?\\)|\\{.*?}|\\[.*?]", "", txt)  # Remove the parameters list
                    rm_parameter_token = rm_parameter.split()
                    method["name"] = rm_parameter_token[1]

                    # Obtain return_type of method
                    if len(rm_parameter_token) == 4:
                        method["return_type"] = rm_parameter_token[-1]
                    elif len(rm_parameter_token) == 2:
                        method["return_type"] = "void"

            methods.append(method)
    else:
        for tr in tables[1].findAll('tr'):
            for td in tr.findAll('td'):
                if "arial italic" in str(td):
                    class_declaration["type"] = class_type[1]
                elif "«interface»" in td.getText():
                    class_declaration["type"] = class_type[2]
                class_declaration["name"] = td.getText().strip()
        class_information["class_declaration"] = class_declaration

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


#  Given a dot file, return the uml dict
def parser_dot(dot_file):
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
    uml_graph = nx.MultiGraph(read_dot(dot_file))

    # Process the nodes to get the non-duplicated nodes
    non_duplicated_node, reduplicated_nodes = merge_reduplicated_nodes(uml_graph)

    uml["nodes"], uml["number_of_nodes"], uml["reduplicated_nodes"] = \
        non_duplicated_node, len(non_duplicated_node), reduplicated_nodes

    # Replacing reduplicated nodes in edge
    old_edges = list(uml_graph.edges)

    # for edge in old_edges:
    #     new_edge = (reduplicated_nodes[edge[0]], reduplicated_nodes[edge[1]])
    #     new_edges.append(new_edge)
    new_edges = [(reduplicated_nodes[edge[0]], reduplicated_nodes[edge[1]]) for edge in old_edges]
    uml["edges"], uml["number_of_edges"] = new_edges, uml_graph.number_of_edges()

    # Extract the node information (label)
    new_nodes_information = {}
    nodes_information = dict(uml_graph.nodes(True))

    for node in non_duplicated_node:
        text = nodes_information[node]["label"]
        # https://beautiful-soup-4.readthedocs.io/en/latest/
        soup = BeautifulSoup(text, features="lxml")
        tables = soup.findAll('table')
        new_nodes_information[node] = parser_node(tables)
    uml["nodes_information"] = new_nodes_information

    # Obtain the edge information (shape or node relationship)
    # new_edges_information = []

    edges_information = list(uml_graph.edges(data=True))
    # for edge in edges_information:
    #     new_edge = [reduplicated_nodes[edge[0]], reduplicated_nodes[edge[1]],
    #                 {"relationtype": get_edge_type(edge[2])}]
    #     new_edges_information.append(new_edge)
    new_edges_information = [[reduplicated_nodes[edge[0]], reduplicated_nodes[edge[1]],
                              {"relationtype": get_edge_type(edge[2])}] for edge in edges_information]
    uml["edge_information"] = new_edges_information

    return uml


# Replace the "#" with "$"
def rewrite_dot_file(dot_file):
    dot_name = dot_file.split(".")[0]
    tem_dot_file = dot_name + "_rewrite" + ".dot"
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


def obtain_uml(input_data_path, sha2repo_path, repo_dir, uml_graph_cmd,
               uml_graph_cmd_opt, dot_path, output_data_dir, save_error_dir, part):

    umls, m2uid, big_graph_id = {}, {}, []
    prev_method_path, prev_repo_sha, prev_package_name, prev_mid = "", "", "", 0
    sha2repository = pickle.load(open(sha2repo_path, "rb"))
    csn_data = pickle.load(open(input_data_path, "rb"))[part]

    for mid, method in csn_data.items():
        repo_sha, method_path = method["sha"], method["path"]
        if method_path == prev_method_path:
            m2uid[mid] = prev_mid
            continue

        repo_name = sha2repository[repo_sha]
        try:
            java_file_path = os.path.join(repo_dir, os.path.normpath(repo_name + "-" + repo_sha + "/" + method_path))
            package_name = get_package_name(java_file_path)
            if prev_repo_sha == repo_sha and prev_package_name == package_name:
                m2uid[mid] = prev_mid
                continue

            # The following code uses uml graph tools to parse package file and get dot file of uml
            # 1. Get package absolute path
            package_path = get_package_path(java_file_path, package_name)
            # 2. Set output_file_path and  output_name
            # dot_file_path = dot_path + str(mid) + ".dot"
            dot_file_path = os.path.join(dot_path, str(mid) + ".dot")
            output_name = uml_graph_cmd_opt + dot_file_path
            # 3.Run the commond line :
            # subprocess.getstatusoutput(uml_graph_cmd + package_path + "  " + package_name + output_name)
            # TODO: why big_graph? continue. # what is 500?
            if get_file_size(dot_file_path) > 500:
                big_graph_id.append(mid)
                continue
            # 4 Replace the "#" with "$" because type of protected in dot is noted # which means comment
            new_dot_file = rewrite_dot_file(dot_file_path)
            print(new_dot_file)
            # 5 Parser information of class from dot
            umls[mid] = parser_dot(new_dot_file)
            m2uid[mid] = mid
            # 6 Record previous method information
            prev_repo_sha, prev_package_name, prev_method_path, prev_mid = repo_sha, package_name, method_path, mid
        except:
            error_file = save_error_dir + "_error.txt"
            traceback.print_exc(file=open(error_file, 'a'))
        if len(umls) > 9:
            break
    # print(m2uid)
    # print(umls)
    save_pickle_data(os.path.join(output_data_dir), 'umls.pkl', umls)
    save_pickle_data(os.path.join(output_data_dir), 'method2uml_index.pkl', m2uid)
    save_pickle_data(os.path.join(output_data_dir), 'big_graph_id.pkl', big_graph_id)


class Config(object):
    """
    attribute:
    [new_data_name, old_data_path, new_data_path,name_list_path, repositories_path , output_path
    uml_graph_cmd, uml_graph_cmd_opt]

    """

    # The path of CSN data
    # https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/java.zip
    # old_data_path = [r"/datadrive/MyCodeSumNet/MethodSum/data/java/java/final/jsonl/train/java_train_" + str(i) +
    #                  ".jsonl " for i in range(16)]
    # old_data_path.append(r"/datadrive/MyCodeSumNet/MethodSum/data/java/java/final/jsonl/valid/java_valid_0.jsonl")
    # old_data_path.append(r"/datadrive/MyCodeSumNet/MethodSum/data/java/java/final/jsonl/test/java_test_0.jsonl")
    # old_data_path = ['/mnt/xiaodi/cocogum_refactor/csn_mini_data/sourceCode/java/final/jsonl/valid/java_valid_0.jsonl']
    # input_data_path = ['/mnt/xiaodi/cocogum_refactor/csn_mini_data/csn.pkl']
    input_data_path = ['../../data/csn/csn.pkl']

    # The path  to save the new data
    # new_data_path = [r"/datadrive/MyCodeSumNet/MethodSum/data/java/java/final/SBT_jsonl/train/" for i in range(16)]
    # new_data_path.append(r"/datadrive/MyCodeSumNet/MethodSum/data/java/java/final/SBT_jsonl/valid/")
    # new_data_path.append(r"/datadrive/MyCodeSumNet/MethodSum/data/java/java/final/SBT_jsonl/test/")
    # output_data_dir = ['/mnt/xiaodi/cocogum_refactor/csn_mini_data/umldata/SBT_jsonl/train/',
    #                    '/mnt/xiaodi/cocogum_refactor/csn_mini_data/umldata/SBT_jsonl/test/',
    #                    '/mnt/xiaodi/cocogum_refactor/csn_mini_data/umldata/SBT_jsonl/valid/']

    output_data_dir = ['../../data/csn/uml/train/',
                       '../../data/csn/uml/test/',
                       '../../data/csn/uml/valid/']

    # name_list.pkl is a dict that saves the  name and sha of  repository
    # {sha: name_of_repository,..... }
    # name_list_path = [r"/datadrive/FunSum/new_data/java/train/name_list.pkl"] * 16
    # name_list_path.append(r"/datadrive/FunSum/new_data/java/valid/name_list.pkl")
    # name_list_path.append(r"/datadrive/FunSum/new_data/java/test/name_list.pkl")
    # name_list_path = ['/mnt/xiaodi/cocogum_refactor/csn_mini_data/sourceCode/github_repositories/valid/name_list.pkl']
    name_list_path = ['../../data/csn/github_repositories/train/sha2repoName.pkl']

    # The path of github repositories which our methods is obtained form
    # repositories_path = [r"/datadrive/FunSum/new_data/java/train/"] * 16
    # repositories_path.append(r"/datadrive/FunSum/new_data/java/valid/")
    # repositories_path.append(r"/datadrive/FunSum/new_data/java/test/")
    # repositories_path = ['/mnt/xiaodi/cocogum_refactor/csn_mini_data/sourceCode/github_repositories/valid/']
    repositories_path = ['../../data/csn/github_repositories/train/']

    # The path to save dot file which is generated by uml graph tools
    # https://www.spinellis.gr/umlgraph/index.html
    # output_path = [r"/datadrive/FunSum/all20200220/dot_ref/train/java_train_" + str(i) + ".jsonl" for i in range(16)]
    # output_path.append(r"/datadrive/FunSum/all20200220/dot_ref/valid/")
    # output_path.append(r"/datadrive/FunSum/all20200220/dot_ref/test/")
    # uml_dot_dir = ["/mnt/xiaodi/cocogum_refactor/csn_mini_data/umldata/dot_ref/train/",
    #                "/mnt/xiaodi/cocogum_refactor/csn_mini_data/umldata/dot_ref/test/",
    #                "/mnt/xiaodi/cocogum_refactor/csn_mini_data/umldata/dot_ref/valid/"]
    uml_dot_dir = ["../../data/csn/uml/dot/train/",
                   "../../data/csn/uml/dot/test/",
                   "../../data/csn/uml/dot/valid/"]

    # The command to generate uml
    # https://www.spinellis.gr/umlgraph/doc/cd-oper.html
    # https://www.spinellis.gr/umlgraph/doc/cd-opt.html
    # Completed command : uml_graph_cmd+ package_path + "  " + package_name +  uml_graph_cmd_opt
    uml_graph_cmd_opt = r" -inferrel  -inferreltype navassoc -collpackages java.util.* -inferdep -inferdepinpackage  " \
                        r" -hide java.* -all -private   -output "
    # uml_graph_cmd = r"java -jar /usr/local/lib/UmlGraph.jar -sourcepath "
    uml_graph_cmd = "java -jar UmlGraph.jar -sourcepath "


if __name__ == '__main__':
    """
    generate train valid test uml files. For example, val set:
    umldata/dot_ref/valid/: uml dot files, 
    umldata/SBT_jsonl/valid/java_val_0.jsonl/
        big_graph_id.pkl
        method2uml_index.pkl
        methods.pkl
        umls.pkl
    """
    start_time = time.perf_counter()

    # Multiprocessing  https://docs.python.org/2/library/multiprocessing.html
    # p = Pool(cpu_count())
    # part = ['train', 'test', 'val']
    part = ['train']

    for i in range(1):
        make_directory(Config.uml_dot_dir[i])
        make_directory(Config.output_data_dir[i])

        ## For jsonl input:
        # p.apply_async(obtain_uml, args=(Config.old_data_path[i], Config.name_list_path[i], Config.repositories_path[i],
        #                                 Config.uml_graph_cmd, Config.uml_graph_cmd_opt, Config.output_path[i],
        #                                 Config.new_data_name[i], Config.new_data_path[i], "", i))

        ## for csn.pkl input:
        # p.apply_async(obtain_uml,
        #               args=(Config.input_data_path[0], Config.name_list_path[0], Config.repositories_path[0],
        #                     Config.uml_graph_cmd, Config.uml_graph_cmd_opt, Config.uml_dot_dir[i],
        #                     Config.output_data_dir[i], "", part[i]))
        obtain_uml(Config.input_data_path[0], Config.name_list_path[0], Config.repositories_path[0],
         Config.uml_graph_cmd, Config.uml_graph_cmd_opt, Config.uml_dot_dir[i],
         Config.output_data_dir[i], "", part[i])

    # p.close()
    # p.join()

    # Print time cost
    end_time = time.perf_counter()
    time_cost = end_time - start_time
    m, s = divmod(time_cost, 60)
    h, m = divmod(m, 60)
    print("time_cost: %02d:%02d:%02d" % (h, m, s))
