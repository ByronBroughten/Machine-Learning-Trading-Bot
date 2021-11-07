import torch, zarr, pickle, os
import datetime as dt
import numpy as np

from IPython.display import display, HTML
from pathlib import Path

import utils


def split_shuffle(data, set_infos, seed):
# takes data of different categories but the same length and splits them into shuffled sets
# as denoted by set_types

    splits = {}

    # dimensions of sets, based on margin of error and confidence interval
    set_lengths = []
    for set_info in set_infos:
        set_lengths.append(set_info['num_examples'])

    # because of manual seed, different groups of data can be split in the same way
    torch.manual_seed(seed)
    split_data = torch.utils.data.random_split(data, set_lengths)
    

    for set_info, a_split in zip(set_infos, split_data):
        splits[set_info['base_name']] = a_split[0:]
        splits[set_info['base_name']] = splits[set_info['base_name']].numpy()
    
    return splits
    

def split_ordered(data, set_infos):
# takes data of different categories but the same length and splits them into unshuffled sets
# as denoted by set_infos

    splits = {}

    index = 0
    for set_info in set_infos:
        splits[set_info['base_name']] = data[index:index+set_info['num_examples']]
        index += set_info['num_examples']
    
    return splits


def split_main(data, set_infos, shuffle, seed=0):
# break data into sets of your liking, pytorch-style
 
    
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data)


    # shuffle when splitting or keep the order
    if shuffle:
        splits = split_shuffle(data, set_infos, seed)
    else:
        splits = split_ordered(data, set_infos)

    
    print('s, v ', end='')
    for s, v in splits.items():
        print(s, v.shape, end = '')
    print()

    return splits