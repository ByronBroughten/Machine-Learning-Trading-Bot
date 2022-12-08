#%% Imports
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
    
# #%%
# import zarr, utils
# test = utils.zarrs.make_array((10, 5), 'test')
# import numpy as np
# test[:] = (test[:] + 1) * np.arange(5)
