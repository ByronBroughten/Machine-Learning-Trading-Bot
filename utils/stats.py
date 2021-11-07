import numpy as np
import scipy.stats as sci_stats
# from statsmodels.tsa.seasonal import seasonal_decompose
# from statsmodels.tsa.stattools import adfuller

def get_sample_size(me, ci, pop=1000000, sd=.5):    
# get statistically significant sample sizes for validation and test sets
# margin of error, confidnece level, standard deviation, population size

    # z score
    z = sci_stats.norm.ppf(1 - (1 - ci) / 2)
    #https://stackoverflow.com/questions/20864847/probability-to-z-score-and-vice-versa-in-python

    # secondary formula for massive populations
    sample_size_big_pop = (z**2 * (sd * (1 - sd))) / me**2
    if pop >= 50000:
        sample_size = sample_size_big_pop
    else:
        # standard formula for smaller populations
        sample_size = sample_size_big_pop / (1 + (z**2 * (sd * (1 - sd))) / (me**2 * pop))
    
    sample_size = int(round(sample_size, 0))
    return sample_size

def get_quartile_from_arrs(arrs, ratio):
    
    total_unique = set()
    num_elements = 0

    # count the elements and get the unique entities for split-cases
    for arr1 in arrs:

        # this kind of stuff is for it to work with zarr arrays
        try: arr[:] = arr1[:]
        except (ValueError, NameError):
            arr = arr1[:]

        if len(arr) > 0:

            total_unique.update(arr)
            num_elements += len(arr)
            
            try: assert upper >= max(arr)
            except (UnboundLocalError, AssertionError):
                upper = max(arr)

            try: assert lower <= min(arr)
            except (UnboundLocalError, AssertionError):
                lower = min(arr)
            
    # below lowest element, above highest, and between each pair
    upper += 1; lower -= 1
    ti = ratio * (num_elements + 1)
    if ratio < .5: ti += .5
    elif ratio > .5: ti -= .5
    print('num_elements', num_elements, 'ratio', ratio, 'ti', ti)

    # pick element on which to base quantile guess
    for _ in range(num_elements):
        print('upper', upper, 'lower', lower)

        unique = set()
        guess = None
        for arr2 in arrs:
            
            try: arr[:] = arr2[:]
            except (ValueError):
                arr = arr2[:]
            
            pool = arr[np.where((arr[:] < upper) & (arr[:] > lower))]
            # print('pool', pool)
            unique.update(pool)
            
            # guess from whichever pools from arrs have values
            if guess is None:
                try: guess = np.random.choice(pool)
                except ValueError: continue
        print('guess', guess) # 'unique', unique

        # criteria for picking winner or updating upper and lower (plus num_left)
        num_below = 0        
        num_guess = 0
        
        num_left = 0

        for arr3 in arrs:
            # all elements less than guess
            try: arr[:] = arr3[:]
            except (ValueError):
                arr = arr3[:]

            sub_arr = arr[np.where(arr[:] < guess)]
            num_below += len(sub_arr)
            
            # all elements that are guess
            num_guess += np.count_nonzero(arr == guess)
            
            # how many elements are being considered at this point
            num_left += len(sub_arr)
        
        print('num_below', num_below, 
                'num_guess', num_guess,
                    'num_left', num_left)
        
        # success conditions
        if ti % 1 == 0:
            tx = ti
            if num_below + 1 == ti or (num_below < ti and num_below + num_guess >= ti):
                print('final guess', guess)
                return guess
        else:
            tj = int(ti)
            tx = tj
        
            # if tj is on both sides of the quartile, tj is it
            if num_below + 1 <= tj and num_below + num_guess > tj:
                print('final guess', guess)
                return guess
            # if tj has the correct number below, then it's time to do some math
            elif num_below + num_guess == tj:
                uq = sorted(list(total_unique))
                inx = uq.index(guess)
                quantile = guess + (uq[inx + 1] - guess) * (ti - tj)
                print('guess, uq[inx + 1], quantile', guess, uq[inx + 1], quantile)
                print('quantile', quantile, '\n')
                return quantile
        
        # conditions to update upper and lower
        if num_below + 1 > tx:
            # too many elements below guess, never guess at or higher again
            upper = guess
        elif num_below + 1 < tx and num_below + num_guess < tx:
            # not enough elements below guess, even with num_guess
            # num_below + num_guess has to be ABOVE ti, or else the next number up
            # will have the ti below it.
            lower = guess
    
    # if you didn't find anything...
    raise ValueError("failed to find ratio point")

def detrend_volume(data):
# https://coderzcolumn.com/tutorials/data-science/how-to-remove-trend-and-seasonality-from-time-series-data-using-python-pandas
# maybe just use linear regression

    seqs = data[['datetime', 'volume']]
    print('len(seqs)', len(seqs))

    
    print('fresh volume')
    vol = seqs['volume']
    show.plot_arr(vol)

    # print('logged seqs')
    # seqs['volume'] += 1
    # seqs['volume'] = seqs["volume"].apply(lambda x : np.log(x))
    # vol = seqs['volume']
    # basic_plot(vol)


    # seqs['datetime'] = np.arange(1, len(seqs)+1)
    # seqs.set_index(['datetime'], drop=True,
    #                             append=False,
    #                             inplace=True,
    #                             verify_integrity=False)
    seqs = seqs.dropna()
    vol = seqs['volume']
    print('min', min(vol))
    print('max', max(vol))
    sub_vol = vol.iloc[500000*2:500000*3]

    dftest = adfuller(sub_vol, autolag = 'AIC')
    print("1. ADF : ",dftest[0])
    print("2. P-Value : ", dftest[1])
    print("3. Num Of Lags : ", dftest[2])
    print("4. Num Of Observations Used For ADF Regression and Critical Values Calculation :", dftest[3])
    print("5. Critical Values :")
    for key, val in dftest[4].items():
        print("\t",key, ": ", val)
    
    # we want p-value to be <= 0.05


    # d_seqs = seqs + .0000000000000000001
    # period=int(len(seqs)/2)
    # decompose_result = seasonal_decompose(d_seqs, model="multiplicative")
    # decompose_result.plot()

    return data

# quartiles_from_arrs_test
# arrs = [np.random.randint(10, size=random.randint(0, 10)),
#         np.random.randint(10, size=random.randint(0, 10))]

# for a, arr in enumerate(arrs):
#     arrs[a] = arrs[a].astype(np.float32)
#     for i in range(len(arrs[a])):
#         arrs[a][i] = arrs[a][i] + round(random.random(), 3)
#         print(arrs[a][i])

# all_arrs = np.concatenate((arrs[0], arrs[1]))
# quantiles = [.25, .5, .75]
# q_guesses = []
# np_guesses = []

# q_range = max(q_guesses) - min(q_guesses)
# print('median_guess', median_guess, 'q_range_guess', q_range)

# both_arrs = both_arrs[:, np.newaxis]
# scaler = RobustScaler().fit(both_arrs)
# print('actual median', scaler.center_, 'scaler q_range', scaler.scale_)


# class Zarr_Loader_Dataset(Dataset):
#     def __init__(self, zarr_data, memory_batch_size):
#         super(Zarr_Loader_Dataset, self).__init__()
#         self.zarr_data = zarr_data
#         self.len = len(self.zarr_data)
#         self.mbs = memory_batch_size

#         self.num_1 = 0
#         self.memory_batch_1 = zarr_data[self.num_1:self.num_1+self.mbs]

#         self.num_2 = self.get_num_2()
#         self.memory_batch_2 = zarr_data[self.num_1:self.num_2]
    
#     def get_num_2(self):
#         num_2 = self.num_1+self.mbs
#         num_2 = num_2 if num_2 <= self.len else self.len - self.num_1
#         return num_2
    
#     def __len__(self):
#         return self.len
    
#     def update_memory_batches(self):
#         self.memory_batch_1 = self.zarr_data[self.num_1:self.num_1+self.mbs]
#         self.num_2 = self.get_num_2()
#         self.memory_batch_2 = self.zarr_data[self.num_2:self.num_2+self.mbs]
    
#     def raise_memory_batches(self, how_many_mbs):
#         self.num_1 += self.mbs * how_many_mbs
#         self.update_memory_batches()
    
#     def lower_memory_batches(self, how_many_mbs):
#         self.num_1 -= self.mbs * how_many_mbs
#         self.update_memory_batches()
        
    
#     def __getitem__(self, index):

#         maxi = self.num_2 + self.mbs - 1
#         if index > maxi:
#             print('raising memory batches')
#             how_many_mbs = math.ceil((maxi - index) / self.mbs)
#             self.raise_memory_batches(how_many_mbs)
#         elif index < self.num_1:
#             print('lowering_memory_batches')
#             how_many_mbs = math.ceil((self.num_1 - index) / self.mbs)
#             self.lower_memory_batches(how_many_mbs)

#         if index < self.num_1 + self.mbs:
#             item = self.memory_batch_1[index-self.num_1]
#         elif index >= self.num_1 + self.mbs:
#             item = self.memory_batch_2[index-self.num_2]

        # # indices_1 = index[np.where(index[:] < self.num_1+self.mbs)] - self.num_1
        # # indices_2 = index[np.where(index[:] >= self.num_1+self.mbs)] - self.num_2
        # # batch_1 = self.memory_batch_1[indices_1]
        # # batch_2 = self.memory_batch_2[indices_2]
        # # batch = np.concatenate((batch_1, batch_2), axis=0)

        # return item


# items = (
#     'batch_size',
#     ('zarr_kwargs', ('dtype', 'chunk_size', 'clevel', 'order')),
#     ('learn', ('every', 'for'))
# )

# def check_reliability(function, param):
#     sizes = []
#     times_taken = []
#     mode, count = 0, 0
#     normal_50 = 50 / param
    
#     for i in range(10):
#         print('*' * 10)
#         print('iteration:', i + 1)
#         size, time_taken = function(param)
        
#         sizes.append(size)
#         times_taken.append(time_taken)

#         mode, count = sci_stats.mode(sizes)
#         std = np.std(np.array(times_taken))
#         mean = np.mean(np.array(times_taken))

#         std_if_50 = normal_50 * std
#         mean_if_50 = normal_50 * mean

        
#         print('mean:', mean)
#         print('batch mode:', mode, 'count:', count)
#         print('time standard deviation:', std)
        
#         print('epochs:', param)
#         print('mean norm to 50:', mean_if_50)
#         print('time standard deviation normalized to 50:', std_if_50)
#         print()
#     return mode, count, std_if_50, mean_if_50
