import itertools, psutil, zarr, numpy as np
from torch.utils.data import Dataset

import utils

def get_XY_dataset(dataset_glob_info_dict):

    XY = {
        'X': [],
        'Y': []
    }
    
    for glob_name, glob_info in dataset_glob_info_dict.items():

        XY[glob_info['X_or_Y']].append({
            'name': glob_name,
            'dataset': glob_info['dataset']
        })
    
    dataset = Variable_Dataset(XY)
    return dataset


class Variable_Dataset(Dataset):
    def __init__(self, XY):
        
        self.XY = XY
        self.num_examples = len(XY['X'][0]['dataset'])

    def __len__(self):
        return self.num_examples
    
    def __getitem__(self, index):
        
        to_return = {}

        for letter, datasets_list in self.XY.items():
            if len(datasets_list) == 1:
                to_return[letter] = datasets_list[0]['dataset'][index]
            
            elif len(datasets_list) > 1:
                to_return[letter] = {dataset_dict['name']: dataset_dict['dataset'][index] for dataset_dict in datasets_list}
        
        return to_return['X'], to_return['Y']
    
    def get_glob(self, dataset_name):
        for datasets_list in self.XY.values():
            for dataset_dict in datasets_list:
                if dataset_dict['name'] == dataset_name:
                    return dataset_dict['dataset']
        else:
            raise ValueError(f'{dataset_name} is not in self.XY')
    
    def writeable(self, writeable=False):
        for datasets_list in self.XY.values():
            for dataset_dict in datasets_list:
                if isinstance(dataset_dict['dataset'], np.ndarray):
                    dataset_dict['dataset'].flags.writeable = writeable
                elif isinstance(dataset_dict['dataset'], zarr.core.Array):
                    dataset_dict['dataset'].read_only = not writeable
                else:
                    raise ValueError(f"dataset_dict['dataset'] is of type {type(dataset_dict['dataset'])}; didn't expect that.")
        

def load_seqs_as_possible(final_set_infos, exclude='train'):
    if exclude is None: exclude = []

    set_names = [set_info['name'] for set_info in final_set_infos if set_info['kind'] != exclude]
    set_name_perms = get_all_permutations(set_names)

    perms_and_nbytes = [get_seq_perm_nbytes(set_name_perm, final_set_infos) for set_name_perm in set_name_perms]
    perms_and_nbytes.sort(key=lambda dic: dic['nbytes'], reverse=True)

    loadable_name_perm = get_biggest_loadable_sets(perms_and_nbytes)

    for i, set_info in enumerate(final_set_infos):
        if set_info['name'] in loadable_name_perm:
            print('loading', set_info['name'], 'seqs into memory')

            for j, glob_dict in enumerate(set_info['dataset'].XY['X']):
                if glob_dict['name'] == 'seqs':
                    
                    glob_dict['dataset'] = utils.zarrs.batch_zarr_to_np(glob_dict['dataset'])
                     # final_set_infos[i]['dataset'].XY['X'][j]['dataset'] = func(set_info['dataset'].get_glob('seqs'))

                    # if we ever switch to using 'split' as a stage and level kind, the second 'dataset' would be 'split'
    
    return final_set_infos

def get_all_permutations(a_list):
    all_combinations = []

    for r in range(len(a_list) + 1):
        combinations_object = itertools.combinations(a_list, r)
        combinations_list = list(combinations_object)
        all_combinations += combinations_list
    
    return all_combinations

def get_seq_perm_nbytes(set_name_perm, final_set_infos):

    nbytes = 0
    for set_info in final_set_infos:
        if set_info['name'] in set_name_perm:
            nbytes += set_info['dataset'].get_glob('seqs').nbytes
    
    return {'nbytes': nbytes, 'set_name_perm': set_name_perm}

def get_biggest_loadable_sets(perms_and_nbytes):
    extra_gigs = 3
    giga_coef = 0.000000001
    giga_avail = psutil.virtual_memory().available * giga_coef - extra_gigs
    
    for pn in perms_and_nbytes:
        if pn['nbytes'] * giga_coef <= giga_avail:
            return pn['set_name_perm']

def sort_by_nbytes_and_set_type(dic, preferred_set=None):
    criteria = dic['nbytes'] * 0.000000001

    if preferred_set is not None:
        if any(preferred_set == a_set[:len(preferred_set)] for a_set in dic['sets']):
            criteria += 100
    
    return criteria