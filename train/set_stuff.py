#%% everything to do with readying sets of data for training
import torch, pickle, zarr, os, shutil
import numpy as np
from torchvision import datasets, transforms
from sklearn.preprocessing import RobustScaler
from torch.utils.data import Dataset
from utils.misc import Objdict

class Func_Dataset(Dataset):
    def __init__(self, data, func):
        super(Func_Dataset, self).__init__()
        self.func = func
        self.data = data
        self.len = len(data)
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):

        data = self.data[index]
        data = self.func(data)
        
        return data

class Single_Varb_Dataset(Dataset):
    def __init__(self, varb):
        print('Initializing single_varb_dataset.')
        if not isinstance(varb, zarr.core.Array):
            try:
                if varb.dtype != np.float32:
                    varb = varb.astype(np.float32)
            except AttributeError:
                raise AttributeError(f"Here's what's causing the problem:\n{varb}")

        self.varb = varb
        self.len = varb.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index): # index was in_index

        # try: # numpy
        #     return self.varb[index]
        # except IndexError: # zarr
        #     return self.varb.oindex[index, :]
    
        return index_np_zarr_etc(self.varb, index)

class Multi_Varb_Dataset():
    def __init__(self, X):
        print('Initializing multi_varb_dataset')
        for x in X:
            if not isinstance(X[x], zarr.core.Array):
                if X[x].dtype != np.float32:
                    X[x] = X[x].astype(np.float32)

        self.multi_x = X
        self.len = next(iter(X.values())).shape[0]
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        # if isinstance(index, torch.Tensor):
        #     index = index.item()

        # should work with numpy, zarr or pytorch arrays
        return {k: index_np_zarr_etc(v, index) for k, v in self.multi_x.items()}

class Env_Multi(Dataset):

    def __init__(self, X, y):

        self.experience_store = Multi_Varb_Dataset(X)
        self.rewards = Single_Varb_Dataset(y)
        self.num_examples = y.shape[0]
            
    # len(dataset)
    def __len__(self):
        return self.num_examples

    def __getitem__(self, index):
        return self.experience_store[index], self.rewards[index]

def index_np_zarr_etc(array, index):
    # wrap outputs in numpy arrays if need be

    try: # numpy
        return array[index]
    except IndexError: # zarr
        return array.oindex[index, :]

def split_X_from_y(sets):
    print('Splitting Xs and ys.')
    data = Objdict()

    for a_set in sets:

        data[a_set] = Objdict()
        data[a_set].X = Objdict()

        for cat in sets[a_set]:
            if cat == 'ys': data[a_set].y = sets[a_set][cat]
            else: data[a_set].X[cat] = sets[a_set][cat]

    return data

def get_trade_env(sets):

    print('Getting trade environment.')
    varbs = split_X_from_y(sets)

    print('varbs\n', varbs)

    print('Creating envs dict.')
    envs = Objdict({a_set: Env_Multi(varbs[a_set].X, varbs[a_set].y) for a_set in varbs})

    return envs

def rm_all_sets(folders_to_clear):
    base_path = "../data/training/chart_equity_subs_mins1"

    for folder in folders_to_clear:
        data_path = os.path.join(base_path, folder)
        set_dirs = os.listdir(data_path)
        for sp in set_dirs:
            set_path = os.path.join(data_path, sp)
            shutil.rmtree(set_path)

# #%%
# import numpy as np, multiprocessing, datetime as dt
# from torch.utils.data import DataLoader, Subset, DataLoader
# import set_stuff

# # from . import utils

# preload = True
# first_load = False
# fl_bsize = 10000

# # if preload:
# #     print('loading')
# #     for phase in analysis:
# #         if phase[:3] == 'val' or phase[:4] == 'test':

# #             if first_load:
# #                 print(analysis[phase]['seqs'].nbytes * 0.000000001)
            

# #             start = dt.datetime.now()
# #             if first_load:
# #                 analysis[phase]['seqs'] = batch_get_seqs(analysis[phase]['seqs'])    
# #                 print(phase, 'preload.', 'batch_size', fl_bsize, dt.datetime.now() - start)
# #             else:
# #                 analysis[phase]['seqs'] = analysis[phase]['seqs'][:]
# #                 print(phase, 'preload.', 'no loader', dt.datetime.now() - start)
            
# full_env = set_stuff.get_trade_env(analysis)

# #%%
# batch_size = 1024

# print('iterating')
# for phase in full_env:
#     if phase[:3] == 'val' or phase[:4] == 'test':
#         env = full_env[phase]
#         indices = np.arange(5) # np.random.permutation(len(env[phase]))
#         subset = Subset(env, indices)
#         dataloader = DataLoader(subset, shuffle=False, batch_size=batch_size,
#                                         num_workers=multiprocessing.cpu_count() - 1)
#         thing = 0
#         start = dt.datetime.now()
#         for X, y in dataloader:
#             for k in X:
#                 print(k)
#                 print(X[k])
#             print('y')
#             print(y)
#         print(phase, 'run. preload', preload, 'batch_size', batch_size,  dt.datetime.now() - start)