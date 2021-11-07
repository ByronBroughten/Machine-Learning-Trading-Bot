#%% Imports

# If I want, I can test big boy 1950 separately after I test the rest. It's just so big that I need
# special parameters to test it. Really, if I'm going to be super serious, I should figure out a function for
# how much space a model will use on the GPU given seq_len, levels, and channels, and maybe the kernel.



#Clean up everything
# Check periodics isn't as efficient as it could be, but run should be good to go. I might need tests for it though.
# Maybe add testing to run.

# test efficacy of moving training window
# implement variable horizons: in this case, I would train n models, with n corresponding to the number of horizons I want
# to use. I would run all those models and get predictions from each about how the price will shift. I would use the loss of those
# model's predictions to rank them from best to worst. A master model will then try to guess all of the losses of all of the horizon
# models, and that prediction will generate an action, picking which horizon model will actually make a trade.
# I can tune each horizon with bayesean optimization

# Get the trader going
# To get a model ready for trading, I must get the most recent kbots data and munge and train with it as usual, with a 
# winning model config, without a test phase, just check that the validation checks out

#Append is Broken
# context (not seqs!) fucked up after I increased train, but not after I increased val. And only precicely
# the last 10000 fucked up, meaning they must have been appended with the wrong data.
# the same thing happened to ys. So it must be an append problem? Or a get_basic problem? But there is no issue with seqs.
# context globs are the right length, but the last 10000 are effed up. Why would it only happen upon increasing train? Seems
# that I was able to add more to val just fine.
# I would guess it has to do with scaling, as only Train seems to have been affected, but since context globs are off,
# this appears to be pre-scaling. y-glob is messed up just like context glob.
# it's gotta be get_basic or append

# 10000 context were appended, but they are about 20 ahead or 40 behind basic
# does append work upsidedown or something? Or is get_basic or something like that the problem?
# they did create and append exactly 10000, though. So they're getting the perfect amount from get_basic

# unrelated--for small append values to work right, I must fix the one hots for context by supplying the full corresponding
# basic columns from which to get the values of the one hots.

#Make Trades
# practice making trades with the api to figure out how getting all the order info works.


#Validity
# in inspect, implement tests for each stage:
# for final you could make sure the sequences relatively match with those of seqs, and that the relevant context values match split context
# for split you could make sure the ys correspond correctly to the closes given the horizons, and you could check that the last
# close of each sequence relatively corresponds to the closes given sequence length
# for glob you could make sure all the closes match exactly

#Dev Env
# use a database instead of a file system
# there should only be a place for temporary zarr arrays

#Accuracy
# use open[i+1] - close[i+horizon] to calculate ys
# that would be the open of the candlestick after the sequence,
# and then minus the close of that candlestick if horizon is 1
# get auto-inspect to work, with and without print


#To verify in munge:
# horizon is not being derived from inside the picture range (if it were then the model would have 100% acc no
# matter what then, though..)
# test doesn't overlap with train or val.

#to deploy simple (this weekend):
# make "basic" with updated kbots data
# set up stream that gets candlesticks that line up with kbots data
# set it up so that if nothing comes through the stream (but the stream is working),
# then it fills
# add candlesticks to basic 

# if not already in a trade,
# make a seq and a context and put them in experience format
# feed into model to make a prediction
# use prediction to go long or short and record what position was taken
# use the horizon of the model to determine when to exit
# repeat.

# get the most recent IBM data from kbots
# train on it with the best model for like 6 epochs
# validate it, I suppose
# let it loose with the above methodology


#Check viability of implementation
# try loading dataframe of trial results
# See whether val score coincides with test score

# train model up to the present

#Improved ML Strategy
# 1. now I have something that's pretty good at predicting the way certain
# horizons will go, based on which I can enter positions.
# 2. now I can make something that uses reinforcement learning to learn
# how to exit positions, whether before or at the horizon
# 3. then, based on how the bot would exit each potential trade and using that
# to create ys, I can use reinforcement learning to learn which trades to enter.

# in relation to the amount of money that can be earned, there
# is a trade-off between smaller horizon and greater accuracy
# that trade-off will be moot if I make a model that can figure out
# the best exit strategy.

#Ultimate Strategy
# Figure out how to best use all streamed quotes rather than candlesticks.

#Use more data
# I want to keep track of the kind of data, standardize it to tda wording
# also the source of the data, such as kbots or stream
# these I can use that information to make h5 or zarr keys 
# I don't have to worry about that quite yet, though

#PBT and Baysean
# get 2TB SSD
# get changing batch sizes mid-run to work
# try combining PBT and Baysean

#If Working on Munging Again
# set up so transfer speed is measured so as is standardized by num examples
# maybe see if append or bigger chunks speeds up chunk transfer

#Speed
# keep data in memory if not switching to a different sequence length
# if I really want to squeek a bit more speed (but sacrifice readability and maintainability),
# make the data loaded into the dataset to be straight torch tensors, and dataset return straight torch tensors
# implement pytorch L2 penalty instead of dropout?

#Test Accuracy
# implement your idea for making train and val sets ongoing and training every
# on them every so often (after determining that val score is indeed related
# to test score). Train and val will comprise a span of time leading up to the 
# present. Here are the three main hyperparameters for this scheme:
# 1. Train and val span (or spans)
# 2. Frequency of re-training/valing
# 3. Batch size for re-training
# I could implement periodic hyperparameter searches, too, though. Hmmm... The
# size of the window to use for validating the hyperparameter search would itself
# be a hyperparameter.

# advanced hyper parameters: possible duplicates per dilation layer, added attention model
# variable channels per layer, context_net and final_net crafting
# mk['final_kwargs'] = {
        #     'input_size': input_size,
        #     'out_size_arr': [input_size]
        # }

#Performance
# maybe figure out a way to calculate what the model size will be based on
# the pertinent parameters.

# what data should I use for this?
# bottlenecks for num channels per layer?
# insert dense layers in TCN?
# restrict actions to long and short?


#Tune
# batch size [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
# dense layer norm [weight norm, batch norm]

#Build
# compare alpha vantage ibm to kbots
# hook it up to an alpha-vantage or kbots datapipeline
# make a better validation set with data from multiple instruments?

#Resources
#free data (but I don't think intraday)
# https://pandas-datareader.readthedocs.io/en/latest/remote_data.html#quandl
# https://www.quandl.com/tools/python
# https://github.com/cuemacro/findatapy
#indicator libs:
# TA-Lib, Tulipy, PyAlgoTrade


# Okay, so the main thing to do is to figure out online learning. The way I see it,
# there are three fundamental perameters:

# to test different methods, I can train (and val I suppose) on half of the data
# to learn features.

# Then I can deploy the same model in a live-mode environment that has the
# parameters: train_interval, memory_size, batch_size (with random sampling)

# baseline is never. train_interval can be based on an if-then statement linked
# to some kind of metric, too. Like maybe the live training could refer to a
# validation set of some kind, and if sufficient over-fitting is detected then
# it knows to not train anymore.


# I could try different sample size parameters to define the number of timesteps
# required to determine whether statistically significant deviation has occured.
# 
# If it has occured, the memory size could go back to whenever was the peak of
# how well the model was performing over any given sample size, from where
# it went down. This will be greater than or equal to the sample size used to
# determine stat sig deviation
# 
# To train the model data could be randomly (or prioritizedly) sampled from the
# memory of that size with whatever batch size is feasible. Will some of the
# memory be used as a validation set? A random sample of the memory will be used
# as a validation set. But this requires the memory to be at least so big as a
# decent validation set can be
#
# the memory size could be 216552. It is then split randomly into two sets of
# 108276, one for train and one for val. Use the model that produces the best
# val score. Perhaps this could be done so often as every night or every weekend.
# If it's going to be on a temporal basis like that, I must keep track of the
# timestamps of ordered data. It can go in a file in the folders of ordered data.

# for online training, ally I'm going to collect and standardize data
# as it comes, store it sequentially, just as this datapipeline allows. But then
# periodically I'm going to break it into sets AFTER it has already been
# standardized.

#Metrics
#the two (or so) things I care about are:

# Macro
# 1. long-term ROI--proportion of long
# Micro (however micro you choose--a month?)
# 1.5 always outperforming long and short--difference is always positive
# 2. risk reduction--fewer span instances under zero given a risk horizon span


#%%
# use more datasets to eliminate if statements and awkward stuff
# make a function for printing run_stage information

from train import run_outer, set_stuff
from utils import analyze
from trade import stream, quick_trade
import utils
def controls(what='train'):
    result = None
    if what == 'train':
        result = run_outer.go()
    if what == 'analyze':
        analysis = analyze.Analysis('/mnt/big_bubba/ray_results_store/phase_two')
        result = analyze.get_analysis_df(analysis)
        utils.show.dataframe(result)
    if what == 'rm_all_sets':
        set_stuff.rm_all_sets(['glob', 'split', 'final'])
        result = 'I removed all of the sets!'
    if what == 'stream':
        stream.stream()
    if what == 'quick_trade':
        result = quick_trade.quick_trade()    
    return result

# tensorboard --logdir=~/jray_results/name
# tensorboard --logdir="/mnt/big_bubba/ray_results_store/phase_one"
# result = controls('rm_all_sets')
result = controls('train')
print(result)
#%%
print('hey ho!')
    
# #%%
# import zarr, utils
# test = utils.zarrs.make_array((10, 5), 'test')
# import numpy as np
# test[:] = (test[:] + 1) * np.arange(5)
