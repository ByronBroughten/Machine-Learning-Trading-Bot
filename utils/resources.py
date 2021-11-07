import zarr, psutil, itertools, shutil, os, numpy as np, multiprocessing, datetime as dt
from pathlib import Path
from torch.utils.data import DataLoader

def get_file_youth(dic):
    return os.path.getmtime(os.path.join(*dic.values()))


def clear_out_fast_drive(path, outer):
    #redone for now to just clear out the fast drive rather than make any transfers
    print('clear_out_fast_drive')
    fast_path = os.path.join(munge.outer_path, 'datasets', munge.folder)
    # big_path = os.path.join(munge.big_outer, 'datasets', munge.data_label)
    
    fast_sets = os.listdir(fast_path)
    # try:
    #     big_sets = os.listdir(big_path)
    # except FileNotFoundError:
    #     print("this path doesn't exist:", big_path)
    #     return

    set_types = tuple(st for sb in munge.set_batches for st in sb['set_types'])

    for a_dir in fast_sets:
        # print('in a_dir')

        fast_seqs_path = os.path.join(fast_path, a_dir, 'seqs')
        Path(fast_seqs_path).mkdir(parents=True, exist_ok=True)
        # print('fast_seqs_path:', fast_seqs_path)

        for a_file in os.listdir(fast_seqs_path):
            # print('in a_file')

            fast_file_path = os.path.join(fast_seqs_path, a_file)
                
            # if it has .tmp, it's empty
            if a_file[-4:] == '.tmp':
                shutil.rmtree(fast_file_path, ignore_errors=True)

            elif a_file != munge.data_of.seqs.file or a_dir not in set_types:
                shutil.rmtree(os.path.join(fast_file_path), ignore_errors=True)
                
                
                # big_seqs_path = os.path.join(big_path, a_dir, 'seqs')
                # # print('big_seqs_path:', big_seqs_path)
                
                # # if the dir and file are in big drive, remove the fast path
                # if a_dir in big_sets and a_file in os.listdir(big_seqs_path):
                #     # print('shutil.rmtree(fast_file_path)')
                #     shutil.rmtree(os.path.join(fast_file_path), ignore_errors=True)
                
                # # or else, transfer the fast path to big
                # else:
                #     zarr_args = munge.data_of.seqs.zarr
                #     big_file_path = os.path.join(big_seqs_path, a_file)

                #     print('fast_file_path', fast_file_path)
                #     fast_stuff = zarr.open(fast_file_path, mode='r')
                #     big_arr = utils.zarr_from_to(fast_stuff, big_file_path, zarr_args['storage'])

                #     # print('remove_fast_file_path')
                #     shutil.rmtree(fast_file_path, ignore_errors=True)

