from util.DataUtil import read_pickle_data, save_pickle_data


def get_dict_n_items(dict, n):
    i = 0
    result = {}
    for k, v in dict.items():
        if i < n:
            result.update({k:v})
            i = i + 1
        else:
            break
    return result


def extract(in_path, out_path_dir, out_filename, n_train=100, n_test=100, n_val=100):
    data = read_pickle_data(in_path)
    result = {}
    result['ctrain'] = get_dict_n_items(data['ctrain'], n_train)
    result['cval'] = get_dict_n_items(data['cval'], n_val)
    result['ctest'] = get_dict_n_items(data['ctest'], n_test)
    result['dtrain'] = get_dict_n_items(data['dtrain'], n_train)
    result['dval'] = get_dict_n_items(data['dval'], n_val)
    result['dtest'] = get_dict_n_items(data['dtest'], n_test)
    result['comstok'] = data['comstok']
    result['datstok'] = data['datstok']
    if 'strain' in data:
        result['strain'] = get_dict_n_items(data['strain'], n_train)
        result['sval'] = get_dict_n_items(data['sval'], n_val)
        result['stest'] = get_dict_n_items(data['stest'], n_test)
        result['smltok'] = data['smltok']
    result['config'] = data['config']
    save_pickle_data(out_path_dir, out_filename, result)

if __name__ == '__main__':
    # data_path = "../data/data_csn_voc1000_funcomformat_200218/standard_data.pkl"
    in_path = "../data/funcom_icse19_remove_dependency/standard_data.pkl"
    out_path_dir = "../data/funcom_icse19_remove_dependency_small_100"
    out_filename = "standard_data.pkl"
    extract(in_path, out_path_dir, out_filename)
