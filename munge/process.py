import zarr, pickle, os, sys, pandas as pd, numpy as np, datetime as dt
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from IPython.display import display

import utils, munge

class Full_Glob_Trimmer(object):
    def __init__(self, seq_len, num_total_final):
        self.seq_len = seq_len
        self.num_total_final = num_total_final
        self.glob_kind = 'basic'
    
    def trim(self, glob):
        if self.glob_kind is None: raise ValueError("Glob trimmer's glob_kind attribute hasn't been set.")
        elif self.glob_kind in ('context', 'price_shift', 'basic'):
            glob = glob[self.seq_len-1:self.seq_len-1+self.num_total_final]
        elif self.glob_kind in ('seqs',):
            pass
        else: raise ValueError(f"Unfamiliar glob_kind of {self.glob_kind}.")
        
        return glob

def get_basic(total_basic, glob_kind, seq_len, num_total_final, num_total_previous):
    # a = np.arange(0, 20); seq_len = 5; num_total_final = 10
    # a[seq_len] == 5 (the 6th element)
    # the first example is at the end of the first sequence: index seq_len-1
    # this can be gotten from a different function based on the type of y or whatever else

    if glob_kind == 'price_shift': # from the first example on to num_total_final + horizon
        basic = total_basic.iloc[:] # seq_len-1:
    elif glob_kind in ('seqs', 'context', 'basic'): # all sequence elements but no horizon elements
        basic = total_basic.iloc[:seq_len-1+num_total_final]
    # elif glob_kind in ('basic'): # just num examples, no unused sequence or horizon elements.
    #     basic = total_basic.iloc[seq_len-1:seq_len-1+num_total_final]
    else:
        raise ValueError(f'Unfamiliar glob_kind of {glob_kind}.')
    
    basic = basic.iloc[num_total_previous:]
    return basic

def drop_cols(df, cols):
    df.drop(columns=cols, inplace=True)
    return df

def df_to_numpy(df, dtype):
    df = df.to_numpy(dtype=dtype)
    return df

def np_change_dtype(arr, dtype):
    arr = arr.astype(dtype)
    return arr

def change_zarr_params(in_arr, at_once, kwargs, outer_path):
    path = f'{outer_path}/tmp/change_zarr_params'
    out_arr = utils.zarrs.make_array(tuple(in_arr.shape), path, **kwargs)

    for i in range(0, len(in_arr), at_once):
        start = dt.datetime.now()
        print(f"{i}", end=" ")
        out_arr[i:i+at_once] = in_arr[i:i+at_once]
        print(dt.datetime.now() - start)

    out_arr = utils.zarrs.replace_arr(in_arr, out_arr)
    return out_arr

def switch_arr_modes(arr, mode):
    arr = zarr.open(arr.store, mode=mode)
    return arr


# seq_len = 3

# seqs = [3, 4, 5, 6, 7, 8, 9, 10]
# ys = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# context = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# so I'm aiming for different amounts of each glob.
# I need as many basics as are necessary to make the right amount of each glob.

# after the globs are big enough, from the others I need to select just seq_len-1:num_needed

# lets say now I want to make 11 and 12
# for seqs I get basic just as I had been.
# I will have a total of 10 seqs

# for the others I will be aiming to have a total of 12
# seq_len - 1 + examples_needed

# so how much basic do I need for them?
# I only need basic[num_pre:]


def scale(dataset, scaler_path, d_format, scalers, is_train, zarr_args=None):
    print('scale path', scaler_path)
    
    scalers = munge.scale.get_what_fit(scalers, dataset, d_format)

    scalers = munge.scale.load_scalers(scalers, scaler_path, d_format, is_train)
    # maybe setpath maker can just create the path that we need and we can use os.path.dirname()

    # fix all the varbs in here
    scalers = munge.scale.make_scalers(scalers, dataset, d_format)
    
    scale_func = munge.scale.Transform(scalers, d_format)

    print("I'm scaling the data.")
    if zarr_args is not None:
        dataset = utils.misc.func_in_chunks(dataset, zarr_args['at_once'], func=scale_func)
    else:    
        dataset = scale_func(dataset)
        
    return dataset

def split(dataset, indices, zarr_args=None):
    print("I'm splitting the data.")

    if zarr_args is None:
        dataset = dataset.iloc[indices]
    else:
        dataset = utils.zarrs.chunk_o_index(dataset, zarr_args['at_once'], indices, zarr_args['outer_path'])
        
    return dataset

def add_axis_if_needed(data, category):
    shapes_to_add_axis = {
        'seqs': 2,
        'context': 1
    }

    if len(data.shape) == shapes_to_add_axis[category]:
        data = data[np.newaxis, :]

    return data

def make_onehots(df, columns):
    onehots = np.array([[]] * len(df))

    for col in columns:
        series = df[col]
        
        unique = series.unique()
        print(unique)

        series = series.to_numpy()
        series = series[:, np.newaxis]
        encoder = OneHotEncoder(sparse=False)
        onehot = encoder.fit_transform(series)

        onehots = np.concatenate((onehots, onehot), axis=1)
    
    return onehots

def make_context(df, features):

    hots = []
    cols = []

    for fi in features:
        hots += fi['onehot']
        cols += fi['not']

    for hot in hots:
        onehots = make_onehots(df, [hot])
        for i in range(onehots.shape[1]):
            name = hot + str(i)
            df[name] = onehots[:, i]
            cols.append(name)

    df = df[cols]
    print('make_context, df.columns', df.columns)
    return df

def get_price_differences(present, previous):

    # the difference in price normalized to a decimal of the preceding price
    # This means that the same price shift will result in a different y from
    # timestep to timestep, because the proportion of price will change.
    reward = (present - previous) / previous
    # as a percentage
    reward = reward * 100
    
    for _ in range(2):
        try:
            rewards = np.zeros((len(reward), 3))
            break
        except TypeError: reward = reward[np.newaxis]
    else: raise Exception('something went wrong making the rewards')

    for _ in range(2):
        try: # for going short
            rewards[:, 1] = -reward
            break
        except ValueError: reward = np.squeeze(reward)
    else: raise Exception('something went wrong making the rewards')
    
    # for going long
    rewards[:, 2] = reward

    # get rid of "-0"
    rewards[rewards==0.] = 0.
    
    return rewards

def price_shift_buy_sell_hold(df, features, horizon, risk_ratio=None, win_loss_ratio=None):
    # remember that when you evaluate ys, you count n=horizon from a starting close,
    # and you're checking whether close[i]'s ys(!!!) sync with close[i] and close[i+horizon]
    
    np_close = df['close'].values[:, np.newaxis]

    previous = np.vstack((np.zeros((horizon, 1)), np_close))
    present = np.vstack((np_close, np.zeros((horizon, 1))))

    rewards = get_price_differences(present, previous)
    
    # drop the first len of horizon cause it's a fraud and you need to in order to align with df.
    rewards = rewards[horizon:]

    # inherit index
    df['hold'] = rewards[:, 0]
    df['short'] = rewards[:, 1]
    df['long'] = rewards[:, 2]

    # dataframe with just ys
    rewards = df[features]

    # drop nonsense last row
    rewards = rewards[:-horizon]

    # sys.exit()
    return rewards


def stand_each_seq(seqs, zarr_args=None):

    if zarr_args:
        seqs = utils.misc.func_in_chunks(seqs, zarr_args['at_once'], stand_each_seq)
    
    else:
        first_points = seqs[:, 0, 0]
        seqs[:, :4, :] = (seqs[:, :4, :].T / first_points).T
         
    return seqs


def make_seqs(df, seq_len, features, zarr_args=None):
    print("I'm making sequences.")

    
    values = df[features].values
    num_chans = len(features)
    # check that column 0 is open and column 4 is volume
    # print('len(df):', len(df))
    # print('values[:5]\n', values[:5])
    # print('df.iloc[:5]\n', df.iloc[:5])

    num_new_seqs = len(df) - seq_len + 1


    if zarr_args is not None:
        
        final_seqs = utils.zarrs.make_array((num_new_seqs, num_chans, seq_len), f"{zarr_args['outer_path']}/tmp/new_seqs.zarr", mode='a',
                                                                    **zarr_args['arr_kwargs'])
        at_once = zarr_args['at_once']
    else:
        at_once = num_new_seqs
    
    start_2 = dt.datetime.now()
    for c in range(0, num_new_seqs, at_once):

        if c + at_once <= num_new_seqs:
            pic_indices = [i for i in range(c, c + at_once)]
        else: pic_indices = [i for i in range(c, num_new_seqs)]

        # make new np_pics if there are none yet or if they're the wrong length
        try:
            assert len(pic_indices) == len(np_pics)
        except (AssertionError, NameError):
            np_pics = np.zeros((len(pic_indices), num_chans, seq_len))
        
        # get a bunch of pics before appending to zarr
        for i, p in enumerate(pic_indices):

            for _ in range(2):
                try:
                    columns[:] = values[p:p+seq_len]
                    break
                except (ValueError, NameError):
                    columns = np.zeros(values[p:p+seq_len].shape)
            else: raise ValueError('Something here went wrong.')

            np_pics[i] = columns.T

        # np_len = len(np_pics)
        # if c % (zarr_at_once * 20) == 0:
        #     utils.plot_ohlcv_seqs(np_pics, (np.random.choice(range(np_len)),))

        if zarr_args is not None:
            final_seqs[c:c+len(np_pics)] = np_pics
        else:
            final_seqs = np_pics
        
        print(f'{c}-{c+at_once}', dt.datetime.now() - start_2)
    

    print('len(df)', len(df))
    print('final_seqs.shape', final_seqs.shape)
    assert num_new_seqs == len(final_seqs)
    
    print(f"I created the sequences, of shape {final_seqs.shape}.")
    return final_seqs


# def make_pics(df, seq_len, category, zarr_at_once=None, zk=None):

#     values = [df[f].values for cf in category['features'] for f in cf['dataset']]
#     num_rows = len(values)

#     num_new_pics = len(df) - seq_len + 1
#     print('num_new_pics', num_new_pics)


#     if os.path.exists(zk.path):
#         zarr_pics = zarr.open(zk.path, mode='a')

#     else:
#         zarr_pics = utils.zarrs.make_array((0, num_rows, seq_len), zk.path, mode='a',
#                                         order=zk.order, chunk=zk.chunk, dtype=zk.dtype,
#                                         clevel=zk.clevel)

#     for c in range(0, num_new_pics, zarr_at_once):
#         start_2 = dt.datetime.now()


#         if c + zarr_at_once <= num_new_pics:
#             pic_indices = [i for i in range(c, c + zarr_at_once)]
#         else: pic_indices = [i for i in range(c, num_new_pics)]        
        

#         np_pics = np.empty((len(pic_indices), num_rows, seq_len))
#         for p, i in zip(pic_indices, range(len(pic_indices))):
            
#             to_stack = [values[l][p:p+seq_len] for l in range(len(values))]
#             np_pics[i] = np.vstack(to_stack)

#         zarr_pics.append(np_pics)

        
#         time_taken_2 = dt.datetime.now() - start_2
#         print('last chunk', time_taken_2)
    
#     print()
#     return zarr_pics

# former context features
# df['price_dif'] = df.high.rolling(seq_len, min_periods=seq_len).max() - \
    #     df.low.rolling(seq_len, min_periods=seq_len).min()

    # df['vol_dif'] = df.volume.rolling(seq_len, min_periods=seq_len).max() - \
    #     df.volume.rolling(seq_len, min_periods=seq_len).min()