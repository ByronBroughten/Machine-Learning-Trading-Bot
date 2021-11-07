import sys, zarr, numpy as np, pandas as pd
from IPython.display import display

import utils

def run_simple_test_packs(arr, varbs, pack_runner, file_type, file_where, fast_packs_only=False):
    
    simple_test_packs = [
        {
            'func': 'munge.inspect.test_dtype',
            'varbs': {'kwargs': {'general': ['dtype']}}
        },{
            'func': 'munge.inspect.test_len',
            'varbs': {
                'args': [
                    {'dataset': ['num_examples']},
                    {'glob': ['kind']}
                ],
                'kwargs': {'general': ['seq_len']}
            }
        },{
            'func': 'munge.inspect.test_not_all_zeros',
            'kwargs': {'file_where': file_where}
        },{
            'func': 'munge.inspect.test_format',
            'kwargs': {
                'file_type': file_type,
                'file_where': file_where
            }
        },{
            'func': 'munge.inspect.test_not_nan',
            'kwargs': {'file_where': file_where}
        }
    ]

    for pack in simple_test_packs:
        if fast_packs_only and pack['func'] in ('munge.inspect.test_not_all_zeros', 'munge.inspect.test_not_nan'):
            continue
        pack_runner(pack, varbs, arr)

# simple test funcs
def test_dtype(arr, dtype):
    try: assert arr.dtype == dtype
    except AssertionError:
        raise TypeError(f"Got array with dtype {arr.dtype} but expected dtype {dtype}.")

def test_len(arr, num_needed, glob_kind, seq_len):
    try: assert len(arr) == num_needed
    except AssertionError:
        raise ValueError(f"Got array with length of {len(arr)} but expected {num_needed}")

    if glob_kind == 'seqs':
        assert arr.shape[-1] == seq_len

def test_not_all_zeros(arr, file_where):

    if file_where == 'in_memory':
        at_once = len(arr)
    elif file_where == 'on_disk':
        at_once = 100000
        # this could be made more nuanced--it could take into account zarr array byte-size or something.
    
    for i in range(0, len(arr), at_once):
        try:
            assert np.any(arr[i:i+at_once] != 0)
            break
        except AssertionError:
            raise ValueError(f"A span of at least {at_once} examples are all zeros.")

    if isinstance(arr, zarr.core.Array):
        assert arr.nchunks_initialized == arr.nchunks
        
def test_format(arr, file_type, file_where):
    asserted = False

    if file_type == 'zarr':
        if file_where == 'on_disk':
            assert isinstance(arr, zarr.core.Array)
            asserted = True
            # return zarr.core.Array
        elif file_where == 'in_memory':
            assert isinstance(arr, np.ndarray)
            asserted = True
            # return np.ndarray
            
    elif file_type == 'h5':
        if file_where == 'in_memory':
            assert isinstance(arr, pd.DataFrame)
            asserted = True
            # return pd.DataFrame
    
    if not asserted: 
        raise ValueError(f'file_where {file_where} and file_type {file_type} combination were not valid.')

def test_not_nan(arr, file_where):
    if file_where == 'in_memory':
        at_once = len(arr)
    
    elif file_where == 'on_disk':
        at_once = 100000
        # this could be made more nuanced--it could take into account zarr array byte-size or something.
    
    for i in range(0, len(arr), at_once):
        assert not np.isnan(np.sum(arr[i:i+at_once]))

## glob and final set tests
def get_relative_greater_or_equal(arr1, arr2=None, displace=1):
    
    if arr2 is None: arr2 = arr1

    relative = np.array([arr1[i] >= arr2[i-displace] for i in range(len(arr1)) if i >= displace])
    return relative

# 2d stuff
def get_glob_col_name(glob_name, col_name):
    glob_col_name = glob_name + '_' + col_name
    return glob_col_name

def get_cols_arr_2d(basic_df, arr_dataset, glob_name, col_num_names):
    for nn in col_num_names:

        glob_col_name = get_glob_col_name(glob_name, nn['name'])
        basic_df.loc[:, glob_col_name] = arr_dataset[:, nn['num']]
    
    return basic_df

def cols_relative_basic(test_glob, glob_name, col_num_names):
    for nn in col_num_names:
        glob_col_name = get_glob_col_name(glob_name, nn['name'])
        
        glob_col = test_glob[glob_col_name].to_numpy()
        basic_col = test_glob[ nn['name']].to_numpy()

        glob_test = get_relative_greater_or_equal(glob_col)
        basic_test = get_relative_greater_or_equal(basic_col)
        comparison = basic_test == glob_test

        try:
            assert comparison.all()
        except AssertionError:
            basic_glob_cols = np.concatenate((basic_col[:, np.newaxis], glob_col[:, np.newaxis]), 1)
            print('\nPrint')
            print('basic_col.shape', basic_col.shape, 'glob_col.shape', glob_col.shape)
            print('basic_col/glob_col\n', basic_glob_cols[:61])
            print(np.where(basic_test[:] != glob_test[:]))
            raise ValueError(f"{glob_col_name} does not match {nn['name']} from basic; see print.")


def get_seq_and_basic_cols(test_glob, col_params, glob_name, offset):
    seq_col_name = get_seq_col_name(glob_name, **col_params['seq_idxs'])

    col = test_glob[seq_col_name].to_numpy()
    basic_col = test_glob[col_params['basic_column']].to_numpy()

    col = col[offset:]
    basic_col = basic_col[:len(basic_col)-offset]

    try:
        assert len(basic_col) == len(col)
    except AssertionError:
        raise ValueError(f'basic_col is length {len(basic_col)} while col is length {len(col)}. offset == {offset}')

    return col, basic_col

def sl_ys_to_close(test_glob, glob_name, col_num_names, horizon, now_col_name, prev_col_name):
    prev = test_glob[prev_col_name].to_numpy()
    now = test_glob[now_col_name].to_numpy()
    # if prev_col_name == 'close' and now_col_name == 'open':
    #     horizon += 1

    col_names = {num_name['name']: get_glob_col_name(glob_name, num_name['name']) for num_name in col_num_names}
    cols = {name: test_glob[col_name].to_numpy() for name, col_name in col_names.items()}

    basic_test = get_relative_greater_or_equal(now, prev, horizon)
    # [now[i] >= prev[i-horizon] for i in range(len(now)) if i >= horizon]

    long_test = np.array([num >= 0 for num in cols['long']])[:-horizon]
    short_test = np.array([num <= 0 for num in cols['short']])[:-horizon]
    # when I shave off the last of these ys, I'm shaving off what basic would need to predict in the future of itself.
    # so this makes sense.

    try:
        comparison = basic_test == long_test
        assert comparison.all()
    except AssertionError:
        print('\nPrint')
        print('basic_test', basic_test.shape, '\n', basic_test[:10], '\n', prev[:10])
        print('long_test', long_test.shape, '\n', long_test[:10], '\n', cols['long'][:10])
        print(np.where(long_test[:] != basic_test[:]))
        raise ValueError('See Print.')
    try: 
        comparison = basic_test == short_test
        assert comparison.all()
    except AssertionError:
        print('\nPrint')
        print('basic_test', basic_test.shape, '\n', basic_test[:10], '\n', prev[:10])
        print('short_test', short_test.shape, '\n', short_test[:10], '\n', cols['short'][:10])
        raise ValueError('See Print.')

    assert all(num == 0 for num in cols['hold'])

def set_span_test(basic_dataset, set_idx, min_basic_idx, num_sb_examples, shuffle):

    wiggle_room = (num_sb_examples // len(basic_dataset)) * 30
        
    if shuffle:
        if wiggle_room > num_sb_examples / 2:
            raise ValueError(f'Set_batch of len {num_sb_examples} is too small to verify with dataset of len {len(basic_dataset)}.')

        assert basic_dataset.index.min() <= set_idx + min_basic_idx + wiggle_room
        assert basic_dataset.index.max() >= set_idx + min_basic_idx + num_sb_examples - wiggle_room
    
    elif not shuffle:
        try:
            assert basic_dataset.index.min() == set_idx + min_basic_idx
        except AssertionError:
                raise ValueError(f"Something's not right: set_idx {set_idx}, min_basic_idx {min_basic_idx}, basic dataset min idx {basic_dataset.index.min()}")

        try:
            assert basic_dataset.index.max() == set_idx + min_basic_idx + num_sb_examples - 1
        except AssertionError:
            raise ValueError(f"Something's not right: set_idx {set_idx}, min_basic_idx {min_basic_idx}, num_sb_examples {num_sb_examples}, basic dataset max idx {basic_dataset.index.max()}")

# seqs
def get_seq_col_name(glob_name, channel, interval):
    seq_column_name = f"{glob_name}_{channel}_{interval}"
    return seq_column_name

def get_seq_col(seqs, glob_name, channel, interval):

    if isinstance(seqs, zarr.core.Array):
        at_once = 100000
    else:
        at_once = len(seqs)
    
    col = np.zeros((len(seqs)))
    
    for i in range(0, len(seqs), at_once):
        col[i:i+at_once] = seqs[i:i+at_once, channel, interval]
        col_name = get_seq_col_name(glob_name, channel, interval)

    return col, col_name

def get_cols_arr_seqs_3d(basic_df, seqs_dataset, glob_name, seq_cols_info):
    for info in seq_cols_info:
        col, col_name = get_seq_col(seqs_dataset, glob_name, **info['made_from']['seq_idxs'])

        basic_df.loc[:, col_name] = col

        if info['standardized_with'] is not None:
            stand_col, stand_name = get_seq_col(seqs_dataset, glob_name, **info['standardized_with']['seq_idxs'])
            if stand_name not in basic_df:
                basic_df.loc[:, stand_name] = stand_col
    
    return basic_df

def test_seqs_3d(test_glob, glob_name, seq_len, seq_cols_info):
    for info in seq_cols_info:

        offset = (seq_len-1) - info['made_from']['seq_idxs']['interval']
        col, basic_col = get_seq_and_basic_cols(test_glob, info['made_from'], glob_name, offset)

        if info['standardized_with'] is None:
            col_test = get_relative_greater_or_equal(col)
            basic_test = get_relative_greater_or_equal(basic_col)

        elif info['standardized_with'] is not None:
            
            stand_col, basic_stand_col = get_seq_and_basic_cols(test_glob, info['standardized_with'], glob_name, offset)
            assert all(num == stand_col[0] for num in stand_col)

            # Here I have a column derived from a channel and interval of seqs. 
            # offset is the difference between (seq_len-1) and interval, and interval can be anywhere from 0 to seq_len-1.
            # a maximum interval means zero offset. 
            # The offset is mostly for basic. After the offset is adjusted for, stand_basic[i-(seq_len-1)] will be what stand_basic[i]
            # must be higher or lower than. While at the same time col is being compared to stand_col. But stand col is probably all
            # one number. I think this will work.

            col_test = get_relative_greater_or_equal(col, stand_col, seq_len-1)
            basic_test = get_relative_greater_or_equal(basic_col, basic_stand_col, seq_len-1)
            # col_test = [col[i] >= stand_col[i - (seq_len-1)] for i in range(len(col)) if i >= seq_len-1]
            # basic_test = [basic_col[i] >= basic_stand_col[i - (seq_len-1)] for i in range(len(basic_col)) if i >= seq_len-1]
        
        comparison = basic_test == col_test
        try:
            assert comparison.all()
        except AssertionError:
            print('\nPrint')
            print('offest:', offset)
            print('seq_cols_info:\n', info)
            print('shapes', 'basic_test', basic_test.shape, ', col_test', col_test.shape)
            basic_vs_col_test = np.concatenate((basic_test[:, np.newaxis], col_test[:, np.newaxis]), 1)
            basic_vs_col = np.concatenate((basic_col[:, np.newaxis], col[:, np.newaxis]), 1)
            if info['standardized_with'] is not None:
                basic_vs_col = np.concatenate((basic_stand_col[:, np.newaxis], basic_vs_col, stand_col[:, np.newaxis]), 1)

            print('basic_test_vs_col_test:')
            print(basic_vs_col_test[:30])
            bad_indices = np.where(basic_test[:] != col_test[:])[0]
            print(bad_indices[:30])
            print('len(bad_indices) vs (max - min)', len(bad_indices), bad_indices[-1] - bad_indices[0])
            print('\nbasic_col_vs_col:')
            print(basic_vs_col[bad_indices[0]:bad_indices[-1]])
            raise ValueError('The test failed. See Print.')