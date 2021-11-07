
import numpy as np
from IPython.display import display, HTML
import matplotlib.pyplot as plt


def dataframe(df):
    display(df)
    # print(df.to_html())
    # display(df.to_html())
    # display(HTML(df.to_html()))

def plot_df(df, cols):
    x = np.arange(len(df))
    plt.plot(x, df[cols].values)
    plt.show()

def plot_arr(arr):
    x = np.arange(len(arr))
    plt.plot(x, arr)
    plt.show()

def print_df_or_np(a_set, indices):
        
    try: # fancy dataframe
        __IPYTHON__
        display(a_set.iloc[indices])
    except (NameError, AttributeError):
        try: # dataframe
            print(a_set.iloc[indices])
        except AttributeError: # numpy array
            print(a_set[indices])

    print()

def picture(pic):
    
    # for zarr
    pic = pic[:]
    
    # for torch or something
    try: pic = pic.numpy()
    except AttributeError: pass

    pic = pic.astype('int32')

    if list(pic.shape) == [3, 224, 224]: pic = np.transpose(pic, (1, 2, 0))
    
    try:
        plt.figure(figsize = (4,4))
        plt.imshow(pic)
        plt.show()
    
    except: (print(pic))

def plot_ohlcv_seqs(seqs, inxs=None):
    
    if inxs is None:
        m = len(seqs)
        inxs = (0, m - 1, np.random.choice(range(m)))

    x = np.arange(seqs.shape[2])

    # plot the seqsuences in inxs, first ohlc, then volume
    if hasattr(seqs, 'oindex'):
        for inx in inxs:
            print('inx,', inx, 'seq\n', seqs.oindex[inx, :, :])
            # ohlc
            plt.plot(x, seqs.oindex[inx, :4, :].T)
            plt.show()
            #volume
            plt.plot(x, seqs.oindex[inx, 4, :].T)
            plt.show()
    else:
        for inx in inxs:
            print('inx', inx, 'seq\n', seqs[inx, :, :])
            # ohlc
            plt.plot(x, seqs[inx, :4, :].T)
            plt.show()
            #volume
            plt.plot(x, seqs[inx, 4, :].T)
            plt.show()

def print_set(a_set, dk, how_much=5, pics=True):
    np.set_printoptions(precision=3, suppress=True)

    print_what = ('ys', 'context', 'seqs')

    if dk in print_what:
        
        shape = a_set.shape
        # for zarr pics we're just grabbing all of them remember
        if dk == 'pics':
            if pics:
                picture(a_set[0])
                picture(a_set[-1])
        
        if dk == 'seqs' and len(shape) == 3:
            m = len(a_set) 
            plot_ohlcv_seqs(a_set, (0, m-1, np.random.choice(range(m))))

        else:
            try: # fancy dataframe
                __IPYTHON__
                display(a_set.iloc[:how_much, :])
                display(a_set.iloc[-how_much:,:])
            except (NameError, AttributeError):
                try: # dataframe
                    print(a_set.iloc[:how_much, :])
                    print(a_set.iloc[-how_much:, :])
                
                except AttributeError: # numpy array
                    print(a_set[:how_much, :])
                    print(a_set[-how_much:, :])
    print()


# #%%
# def show_candlesticks(data=None, min_per_stick=1):
# from mpl_finance import candlestick_ohlc
# import matplotlib.dates as mpl_dates

#     label = 'IBM_adj_1998_min1'

#     if data is None:
#         data = pd.read_hdf(f"../../dataglobs/basic/{label}")
    
#     if min_per_stick > 1:
#         data = one_min_to_more_ohlc(data, min_per_stick)
#         print(data.iloc[0:4])
    
#     # data['datetime'] = data['datetime'].apply(dt.datetime.utcfromtimestamp)
    
#     plt.style.use('ggplot')

#     # Extracting Data for plotting
#     ohlc = data.loc[:, ['datetime', 'open', 'high', 'low', 'close']]
#     ohlc['datetime'] = pd.to_datetime(ohlc['datetime'])
#     ohlc['datetime'] = ohlc['datetime'].apply(mpl_dates.date2num)
#     ohlc = ohlc.astype(float)

#     # Creating Subplots
#     fig, ax = plt.subplots()

#     candlestick_ohlc(ax, ohlc.values, width=0.6, colorup='green', colordown='red', alpha=0.8)
#     print('candlestick_ohlc here')

#     # Setting labels & titles
#     ax.set_xlabel('Date')
#     ax.set_ylabel('Price')
#     fig.suptitle(f'{min_per_stick} Minute Candlestick Chart of {label}')

#     # Formatting Date
#     date_format = mpl_dates.DateFormatter('%d-%m-%Y')
#     ax.xaxis.set_major_formatter(date_format)
#     fig.autofmt_xdate()
#     fig.tight_layout()

#     print('date formatting stuff here')

#     plt.show()