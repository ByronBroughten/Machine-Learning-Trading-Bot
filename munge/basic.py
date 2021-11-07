import os, pandas as pd, datetime as dt, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import display, HTML

import utils, munge
pack_runner = utils.misc.Pack_Runner(globals())

def one_min_to_more_ohlc(df, min_per_stick):

    # df.set_index('datetime', inplace=True)
    df['datetime'] = pd.to_datetime(df['datetime'], unit='s')
    
    # how to create new candlesticks
    ohlc_dict = {                                                                                                             
        'open':'first',                                                                                                    
        'high':'max',                                                                                                       
        'low':'min',                                                                                                        
        'close': 'last',                                                                                                    
        'volume': 'sum'
    }
    # everything else is taken from the same index as close
    for cl in list(df.columns):
        if cl not in ohlc_dict: ohlc_dict[cl] = 'last'

    df = df.resample(f'{min_per_stick}T', closed='left', label='left',
                        on='datetime').apply(ohlc_dict)
    
    del df['datetime']
    df = df.dropna()

    return df

def get_seasonality(df):
    #%% get the seasonal variables

    # Weekday, hour, and minute are for now good. If I use a dataset spanning enough years, 
    # day of year and holidays could be good.

    temp_df = pd.DataFrame().reindex_like(df)
    # I could probably do this without creating a whole nother data frame.

    #get the readable datetime
    temp_df['utc_datetime'] = [dt.datetime.utcfromtimestamp(t) for t in df['datetime']]
    #df['datetime'].apply(lambda t: dt.datetime.utcfromtimestamp(t))
    
    # create the seasonal variables
    df['weekday'] = [t.weekday() for t in temp_df['utc_datetime']]
    #temp_df['utc_datetime'].apply(lambda t: t.weekday())

    df['hour'] = [t.hour for t in temp_df['utc_datetime']]
    #temp_df['utc_datetime'].apply(lambda t: t.hour)
    df['minute'] = [t.minute for t in temp_df['utc_datetime']]
    #temp_df['utc_datetime'].apply(lambda t: t.minute)
    return df

# Determine how many time gaps there are in the data and how big they are.
def get_seconds_frequency_and_timegaps(df):
    # get the difference in time between timestamps

    # line up each next timestamp with its reference timestamp:
    # 1 2 3 4 5 0 From each candle's next unit of the day
    # 0 1 2 3 4 5 minus the present unit of the day
    focal_times = df.iloc[0:]['datetime'].to_numpy()
    next_times = np.hstack((focal_times, np.zeros((1)))).astype(int)
    focal_times = np.hstack((np.zeros((1)), focal_times)).astype(int)

    # subtract for the differences in time
    gaps_next = (next_times - focal_times).astype(int)

    # get rid of the nonsense differences derived by subtracting zero from the first datetime and
    # by subtracting the last datetime from zero
    gaps_next = gaps_next[1:-1]

    # get the closest difference between any of the timestamps, in seconds
    seconds_frequency = np.min(gaps_next).astype(int)
    # For now this will only work if there are no duplicate timestamps.

    # standardize the time gaps by that difference
    gaps_next = ((gaps_next / seconds_frequency) - 1).astype(int)

    # We'll need that seconds_frequency later, too.
    return gaps_next, seconds_frequency

def drop_non_trading_rows(df):

    # only include rows such that...
    #df = df[(((df.hour <= 16) & (df.hour >= 10)) | ((df.hour == 9 & df.minute >= 30)))]
    
    i = df[(df.hour > 15)].index
    df = df.drop(i)

    i = df[(df.hour < 9)].index
    df = df.drop(i)

    i = df[((df.hour == 9) & (df.minute < 30))].index
    df = df.drop(i)

    # experimental
    i = df[(df.weekday > 4)].index
    df = df.drop(i)

    return df

def prep(raw_path):
    step_mins = 1

    #%% for working on it:
    #label = 'IBM_adj_1998_min1'

    df = pd.read_hdf(raw_path)

    # standardizes 'datetime' timestamp to seconds
    # df['datetime'] = utils.misc.stamp_mils_to_secs(df['datetime'], div=100)
    # print(df['datetime'])

    #%% get seconds frequency, no need to worry about gaps for now
    gaps_next, seconds_frequency = get_seconds_frequency_and_timegaps(df)
    assert seconds_frequency == 60

    #%% extend drop trading rows to get rid of weekend days
    mini = df['datetime'].min(); maxi = df['datetime'].max()
    new_index = pd.Index(np.arange(mini,maxi,seconds_frequency), name="datetime")
    df = df.set_index("datetime").reindex(new_index).reset_index()

    #%% get time information 
    df = get_seasonality(df)

    #%% drop rows that are outside of trading hours
    df = drop_non_trading_rows(df)
    #display(HTML(df.iloc[0:100, :].to_html()))

    #%% 
    # index tricks to get new rows
    # replace nans in volume with zeros
    # replace nans in close with ffill
    # replace nans in others with close
    df.volume.fillna(0, inplace=True)
    df.close.fillna(method='ffill', inplace=True)
    df.high.fillna(df.close, inplace=True)
    df.low.fillna(df.close, inplace=True)
    df.open.fillna(df.close, inplace=True)
    df.datetime = df.datetime.astype('int32')

    if step_mins > 1:
        df = features.prep.one_min_to_more_ohlc(df, step_mins)
    
    df.index = np.arange(len(df))

    return df

def make_basic(basic_save_or_append, varbs, raw_path):
    print("I'm making the basic data.")
    start = dt.datetime.now()

    basic = prep(raw_path)

    pack_runner(basic_save_or_append, varbs, basic)
    # basic.to_hdf(basic_path, key='df', mode='w')

    print(f"I made and saved the basic data, which took this long: {dt.datetime.now() - start}")
    print('basic.shape', basic.shape)
    display(basic.iloc[0:3, :]); display(basic.iloc[-3:, :])

def get_total_basic(seq_len, num_examples, horizon, basic_access, varbs, raw_path=None):
    total_needed = seq_len-1+num_examples+horizon

    for _ in range(2):
        all_basic = pack_runner(basic_access['load'], varbs)
        if len(all_basic) > 0:
            break
        else:
            if raw_path:
                varbs['general']['num_basic_previous'] = len(all_basic)
                make_basic(basic_access['save_or_append'], varbs, raw_path)
            else:
                raise FileNotFoundError(f"No basic data found at {basic_path}, and you didn't provide a raw_path.")

    basic = all_basic.iloc[:total_needed]
    if len(basic) == total_needed:
        return basic
    elif len(basic) < total_needed:
        raise FileNotFoundError(f"I'm sorry, there's only {len(basic)} basic samples but you asked for {total_needed}.")
    else:
        raise FileNotFoundError(f"I'm not sure what happened. \
                                The basic data has a {len(basic)} examples, but {total_needed} are needed")