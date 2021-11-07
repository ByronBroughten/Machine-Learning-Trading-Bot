import zarr, os, multiprocessing, shutil
import datetime as dt, numpy as np

from torch.utils.data import DataLoader, Subset, Dataset
from pathlib import Path
from numcodecs import blosc, Blosc

import utils


def get_store(path):
    store = zarr.NestedDirectoryStore(path)
    return store

def get_copy(in_arr, path, tmp=False):
        out_store = get_store(path)
        zarr.copy_store(in_arr.store, out_store)
        out_arr = zarr.open(out_store, mode='r+')
        return out_arr

def make_array(shape, path, mode='w', order='C', chunk=1, dtype='f4', clevel=5, store=None):
    if isinstance(chunk, int): chunk = tuple(chunk if i == 0 else None for i in range(len(shape)))

    store = get_store(path)
    
    arr = zarr.open_array(store, shape=shape, mode=mode, order=order,
    chunks=chunk, dtype=dtype, fill_value=0,
    compressor=Blosc(cname='zstd', clevel=clevel, shuffle=Blosc.SHUFFLE))

    return arr

def get_empty_like(arr_in, out_path, new_len=None):
    print('In utils.zarrs.empty_like.')
    print('arr_in.shape', arr_in.shape)

    out_shape = list(arr_in.shape)
    if new_len is not None:
        out_shape[0] = new_len
    
    print('out_shape', out_shape)

    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    out_store = get_store(out_path)
    out_arr = zarr.creation.open_like(arr_in, out_store, shape=out_shape)
    print('got an empty array with shape', out_arr.shape)

    return out_arr

def arr_to_parallel_loader(arr, batch_size, indices=None):
    if indices is not None:
        arr = Subset(arr, indices)

    parallel_loader = DataLoader(arr,
                        shuffle=False,
                        batch_size=batch_size,
                        num_workers=multiprocessing.cpu_count() - 1)
    return parallel_loader

def batch_zarr_to_np(zarr_arr):
# This is so that the transfer doesn't use too much memory at any one time.
    batch_size = 10000
    first_loader = DataLoader(zarr_arr, shuffle=False,
                                        batch_size=batch_size,
                                        num_workers=multiprocessing.cpu_count() - 1)

    np_arr = np.zeros(zarr_arr.shape, dtype=zarr_arr.dtype)
    indices = np.arange(0, len(np_arr), batch_size)
    for sub_arr, index in zip(first_loader, indices):
        np_arr[index:index+len(sub_arr)] = sub_arr

    return np_arr

def parallel_func_n_chunk(arr, at_once, func):

    parallel_loader = arr_to_parallel_loader(arr, at_once)
    loop_indices = range(0, len(arr), at_once)
    
    for batch, li in zip(parallel_loader, loop_indices):
        
        print('batch', li, end=' ')
        try: print(li+len(batch), 'loader', dt.datetime.now()-start3, end='')
        except UnboundLocalError: pass

        start1 = dt.datetime.now()
        batch = func(batch.numpy())
        print(' func', dt.datetime.now()-start1, end='')

        start2 = dt.datetime.now()
        arr[li:li+len(batch)] = batch # arr_out.append(chunk)
        print(' transfer', dt.datetime.now()-start2, end='')
        
        try: print( ' total', dt.datetime.now() - start3)
        except UnboundLocalError: print()
        start3 = dt.datetime.now()
    
    return arr

# def print_zarr_stats(arr):
#     print('dtype', arr.dtype)
#     print(arr.store.path)

def chunk_o_index(arr_in, at_once, ordered_indices, outer_path):
    tmp_path = f'{outer_path}/tmp/chunk_o_index.zarr'
    arr_out = get_empty_like(arr_in, tmp_path, new_len=len(ordered_indices))

    ordered_indices = np.array(ordered_indices)

    batch_size = at_once
    loop_indices = np.array([i for i in range(0, len(arr_in), batch_size)])

    print(f"I'm o-indexing {len(ordered_indices)} pieces out of {len(arr_in)}")
    start1 = dt.datetime.now()

    num_indexed = 0
    for li in loop_indices:
        start = dt.datetime.now()
        print(li, end=" ")
        
        batch_arr_in = arr_in[li:li+batch_size]
        indices = ordered_indices[np.where((ordered_indices[:] >= li) \
                                                & (ordered_indices[:] < li + len(batch_arr_in)))]

        if len(indices) == 0: continue # no indexes in this outer batch
        indices -= li # line up the indexes with the current batch
        arr_out[num_indexed:num_indexed+len(indices)] = batch_arr_in[indices] #batch_arr_in.oindex[indices]

        num_indexed += len(indices)
        print(dt.datetime.now() - start)
    
    arr_out = replace_arr(arr_in, arr_out)
    print('Total o_index time:', dt.datetime.now() - start1)
    return arr_out

def replace_arr(arr_to_del, arr_to_rep):
# take one arr, delete it, and put the other in its place
    rep_path = arr_to_del.store.path
    shutil.rmtree(rep_path, ignore_errors=True)
    arr_to_rep = rename_arr(arr_to_rep, rep_path)
    return arr_to_rep

def rename_arr(arr, new_path, mode='r+'):
    os.rename(arr.store.path, new_path)
    store = get_store(new_path)
    arr = zarr.open(store, mode=mode)
    return arr

def try_load(path):
    try: arr = zarr.open(path, mode='r+')
    except ValueError: arr = None
    return arr

def get_sorted_paths_by_age(outer_path, data_cat):
    Path(outer_path).mkdir(parents=True, exist_ok=True)
    paths = [{
                'outer': outer_path,
                'middle': os.path.join(d_label, a_set, data_cat),
                'file': a_cfg}\
                    for d_label in os.listdir(outer_path)\
                    for a_set in os.listdir(os.path.join(outer_path, d_label))\
                    for a_cfg in os.listdir(os.path.join(outer_path, d_label, a_set, data_cat))]
    paths.sort(key=utils.resources.get_file_youth)
    return paths

def check_disks_zarr(fast_outer, big_outer, shared, dk, main_file, zarr_args):

    fast_device_path = '/'
    big_device_path = '/mnt/big_bubba'

    fast_outer = os.path.join(fast_outer, 'datasets')
    big_outer = os.path.join(big_outer, 'datasets')

    # if the file is already on the fast, then great
    fast_path = os.path.join(fast_outer, shared, dk, main_file)
    print('fast_outer, shared, dk, main_file\n',
    fast_outer, shared, dk, main_file)
    print('checking', fast_path)
    try:
        fast_stuff = zarr.open(fast_path, mode='r')
        print('got', os.path.basename(fast_path))
        return fast_stuff
    except ValueError: pass

    fast_free_needed = 0
    big_free_needed = .2
    # what if a set from this bunch pushes things well over the edge but then needs a val
    # set to accompany it?
    
    # otherwise, we gotta move it from elsewhere or make it, and maybe make space, too
    big_path = os.path.join(big_outer, shared, dk, main_file)
    fast = shutil.disk_usage(fast_device_path)
    if fast.free / fast.total > fast_free_needed:
        # we don't have to make space for it, so we don't have to delete anything
        fast_paths = []
    else:
        # we'll have to loop through and move or delete the oldest stuff as necessary
        fast_paths = get_sorted_paths_by_age(fast_outer, dk)

    for _ in range(len(fast_paths) + 1):
        try:
            # see if there's enough room in the fast for files
            fast = shutil.disk_usage(fast_device_path)
            assert fast.free / fast.total > fast_free_needed

            # if we can just get it from big, then great.
            # otherwise it returns None and the file will be made
            
            try: big_stuff = zarr.open(big_path, mode='r')
            except ValueError: return None

            fast_stuff = zarr_from_to(big_stuff, fast_path, zarr_args['datasets'], tmp=False)
            return fast_stuff

        # not enough room in fast, gotta move something from fast to big
        except AssertionError:
            print('we in big files')
            
            # if big has room, no need to loop through it to delete stuff
            big = shutil.disk_usage(big_device_path)
            if big.free / big.total > big_free_needed:
                big_paths = []
            else:
                # otherwise, we gotta loop to delete the oldest stuff to make room
                big_paths = get_sorted_paths_by_age(big_outer, dk)
                    
            for __ in range(len(big_paths) + 1):
                try:
                    # see if there's room to move files to big
                    big = shutil.disk_usage(big_device_path)
                    assert big.free / big.total > big_free_needed

                    # move the oldest relevant file in fast to big
                    from_path = os.path.join(fast_outer, fast_paths[0]['middle'], fast_paths[0]['file'])
                    to_path = os.path.join(big_outer, fast_paths[0]['middle'], fast_paths.pop(0)['file'])
                    
                    print('fast to big')
                    from_stuff = zarr.open(from_stuff, mode='r')
                    big_stuff = utils.zarr_from_to(from_path, to_path, zarr_args['storage'])

                    shutil.rmtree(from_path, ignore_errors=True)
                    print('moved')
                
                # not enough room in big, gotta get rid of something, end of the line
                except AssertionError:
                    shutil.rmtree(os.path.join(big_paths[0]['middle'], big_paths.pop(0)['file']),
                                    ignore_errors=True)

# def try_load_into_memory 
    # get from_stuff ready
    # print('trying to load into memory'); start0 = dt.datetime.now()
    # try:
    #     from_batch = from_stuff[:]
    #     mb_size = len(from_stuff)
    #     mb_indices = np.array([0])
    #     print('loaded', dt.datetime.now() - start0)
    # except MemoryError:
    #     print(f"couldn't load from_stuff of shape {from_stuff.shape} bytes {from_stuff.nbytes * 0.000000001} into memory")