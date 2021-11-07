import os, sys, copy, shutil, pandas as pd
import munge, utils


# okay so, I think Im just gonna make a key from scratch
def get_training_key(path, label_beginning):

    # if mode == 'dir':
    labels = os.listdir(path)
    # elif mode == 'hdf':
    #     with pd.HDFStore(path) as hdf:
    #         labels = utils.misc.str_lst_drop_char(hdf.keys(), '/')

    winners = [label for label in labels if label[:len(label_beginning)] == label_beginning]
    
    if len(winners) > 1: raise ValueError(f'There is more than one entry in {path} that starts with {label_beginning}.')
    elif len(winners) < 1: raise FileNotFoundError(f"I didn't find an item that starts with {label_beginning} in {path}.")
    elif len(winners) == 1:
        return winners[0]

def del_tmps(outer_path):
    print(f"I'm deleting everything in {outer_path}/tmp.")
    tmps_dir = os.path.join(outer_path, 'tmp')
    tmps = os.listdir(tmps_dir)
    for tmp in tmps:
        tmp_path = os.path.join(tmps_dir, tmp)
        shutil.rmtree(tmp_path)

def get_path_pieces(outer_path, data_path, price_data_info):
    folder = get_tda_datatype_dir(**price_data_info['training_and_trading'])
    data_outer = os.path.join(outer_path, data_path, folder)

    file_key = get_symbol_key(**price_data_info['training_only'])

    return data_outer, file_key

def get_tda_datatype_dir(tda_datatype, step_mins):
    folder = f"{tda_datatype}_mins{step_mins}"
    return folder

def get_symbol_key(symbol, vendor=None):
    key = symbol
    if vendor:
        key += '_' + vendor
    return key

def get_datapath(data_outer, data_stage, mid_path, file_name, file_key, file_type):
    # requires that this gets the key at some point
    full_name = f"{file_name}-{file_key}"
    full_name += '.' + file_type

    data_path = os.path.join(data_outer, data_stage, mid_path, full_name)
    return data_path

    
def get_scaler_path(outer, glob_name, file_key, train_name):
    scaler_path = os.path.join(outer, 'scalers', f"{glob_name}-{file_key}-{train_name}")
    return scaler_path


def init_training_munge(data_outer, file_key):
    raw_outer_path = os.path.join(data_outer, 'raw')
    file_key = get_training_key(raw_outer_path, file_key)
    raw_path = os.path.join(raw_outer_path, file_key)
    
    return file_key, raw_path