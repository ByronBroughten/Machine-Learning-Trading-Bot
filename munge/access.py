#%% get all of the data ready

# waiting to see how viable google colab is going to be--this might all be moot if I end up using another
# cloud service. Also curious about the speed of going through an epoch.

import os, copy, zarr, pandas as pd, numpy as np
import utils, munge
from pathlib import Path

# path then extant, to mirror zarr.load()
def zarr_cold_append(path, extant):
    store = utils.zarrs.get_store(path)
    arr = zarr.open(store, mode='r+')
    arr.append(extant)
    return arr

def zarr_cold_save(path, extant):
    extant = utils.zarrs.rename_arr(extant, path)
    return extant

def zarr_cold_load(path):
    store = utils.zarrs.get_store(path)
    arr = zarr.open(store, mode='r+')
    return arr

def zarr_warm_append(path, extant):
    store = utils.zarrs.get_store(path)
    arr = zarr.open(store, mode='r+')
    arr.append(extant)
    extant = arr[:]
    return extant

def zarr_warm_save(path, extant):
    zarr.save(path, extant)
    return extant

def load_hdf5(path):
    loaded = pd.read_hdf(path, key='df', mode='r+')
    return loaded

def save_hdf5(path, extant):
    extant.to_hdf(path, key='df', mode='w', format='table')
    return extant

def append_hdf5(path, extant):
    for i in range(2):
        try:
            extant.to_hdf(path, key='df', mode='r+', append=True, format='table')
        except ValueError:
            if i == 0:
                old = pd.read_hdf(path, mode='r+')
                for col in old.columns:
                    if col not in extant.columns:
                        if all(value in (0, 1) for value in old[col]):
                            extant[col] = np.zeros((len(extant)))
                # this is probably fucking up context, but I can't think of why it would fuck up price-shift.
                # what can I do about it fucking up context?

            elif i == 1:
                print('\nPrint')
                print('to append; shape, columns', extant.shape, extant.columns)
                old = pd.read_hdf(path, mode='r+')
                print('old; shape, columns', old.shape, old.columns)
                raise ValueError('Tried to append incompatible columns, see Print.')

    extant = pd.read_hdf(path, mode='r+')
    return extant

def get_access_func(mode, file_type, file_where):
    access_options = {
        'zarr': {
            'in_memory': {
                'load': zarr.load,
                'save': zarr_warm_save,
                'append': zarr_warm_append,
            },
            'on_disk' : {
                'load': zarr_cold_load,
                'save': zarr_cold_save,
                'append': zarr_cold_append,
            }
        },
        'h5': {
            'in_memory': {
                'load': load_hdf5,
                'save': save_hdf5,
                'append': append_hdf5
            }
        }
    }
    return access_options[file_type][file_where][mode]

def try_load(path, file_type, file_where):
    print(f"\nTrying to load from {path}")

    load_func = get_access_func('load', file_type, file_where)

    try:
        loaded = load_func(path)
        assert loaded is not None
    except (FileNotFoundError, ValueError, AssertionError):
        print(f"I failed to load data, so here's an empty array.")
        loaded = []
    else:
        print("No loading error.")

    if isinstance(loaded, zarr.core.Array):
        assert loaded.store.path == path
    
    return loaded

def save(dataset, file_type, file_where, path):
    # print("I'm in access.save.")

    save_func = get_access_func('save', file_type, file_where)
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    dataset = save_func(path, dataset)

    if isinstance(dataset, zarr.core.Array):
        assert dataset.store.path == path

    return dataset

def append(dataset, path, num_total_previous, file_type, file_where):

    if num_total_previous == 0:
        print('Saving rather than appending.')
        dataset = save(dataset, file_type, file_where, path)
    
    else:
        append_func = get_access_func('append', file_type, file_where)
        dataset = append_func(path, dataset)
    
    if isinstance(dataset, zarr.core.Array):
        assert dataset.store.path == path
    
    return dataset


# def append_in_memory(data_one, data_two):
#     if isinstance(data_one, pd.DataFrame):
#         data_one = data_one.append(data_two)
#     elif isinstance(data_one, zarr.core.Array):
#         data_one.append(data_two)
#     elif isinstance(data_one, np.ndarray):
#         data_one = np.concatenate((data_one, data_two))
#     else:
#         raise ValueError('What kind of data is this?\n{data_one}')

#     return data_one