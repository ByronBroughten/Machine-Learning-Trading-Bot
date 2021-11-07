import torch, random, time, copy, multiprocessing
import numpy as np
import datetime as dt

from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from ray import tune

import utils.misc as utils
#import torch_xla, torch_xla.core.xla_model as xm

class Info(object):
    def __init__(self, set_info, iter_info, epoch_info):
        self.levels = ('set', 'iter', 'epoch')
        self.set = set_info
        self.iter = iter_info
        self.epoch = epoch_info
    
    def examples_ran(self, level):
        num_examples = getattr(self, level)['num_examples_ran']
        # if level == 'set':
        #     num_examples = getattr(self, level)['num_examples_ran'] + self.epoch['num_examples_ran']
        # elif level in ('epoch', 'iter'):
        #     num_examples = getattr(self, level)['num_examples_ran']
        # else: raise ValueError("Received level variable not in ('set', 'iter', 'epoch')")
        return num_examples
    
    def batches_ran(self, level):
        num_batches = getattr(self, level)['num_batches_ran']
        # if level == 'set':
        #     num_batches = getattr(self, level)['num_batches_ran'] + self.epoch['num_batches_ran']
        # elif level in ('epoch', 'iter'):
        #     num_batches = getattr(self, level)['num_batches_ran']
        return num_batches 

    def epochs_ran(self, level):
        num_epochs = getattr(self, level)['num_epochs_ran']

        if level == 'set':
            num_epochs += self.epoch['num_examples_ran'] / self.set['num_examples']
        elif level == 'iter':
            num_epochs += (self.iter['num_examples_ran'] - self.set['num_examples'] * num_epochs)/ self.set['num_examples']

        # if level == 'set':
        #     num_epochs = getattr(self, level)['num_epochs_ran'] + self.epochs_ran('epoch')
        # elif level == 'iter':
        #     straight_epochs = getattr(self, level)['num_epochs_ran']
        #     fractional_epochs = (self.examples_ran('iter') - self.set['num_examples'] * straight_epochs) / self.set['num_examples']
        #     num_epochs = straight_epochs + fractional_epochs

        # elif level == 'epoch':
        #     num_epochs = self.epoch['num_examples_ran'] / self.set['num_examples']
        return num_epochs
    
    def print_stage_stats(self, stage):
        print(stage, end=" ")
        for metric in ('time', 'epochs', 'batches', 'examples'):
            if metric == 'time':
                print(getattr(self, stage)['time_ran'], end=", ")
            else:
                print(f"{metric}: {getattr(self, f'{metric}_ran')(stage)}", end=", ")
        print()

# it's running half an epoch
# the half epoch gets added to iter, the epoch is not reset
# epoch retains its value, and so when epochs iter is assessed in periodic check, epochs iter shows that the 
# so when getting iter stats, iter can't just borrow from epochs, because epochs got to where it was 
# possibly in the PREVIOUS iter.
# so iter's examples, and batches for that matter, must be added to independently, and it can't derive its epoch value from 
# a fraction of epoch_info. As its epochs are added to independently


def get_run_stage_info(run_stage, info=None):

    if info is None: info = {}

    info.update ({
        'run_stage': run_stage,
        'num_examples_ran': 0,
        'num_batches_ran': 0,
        'time_ran': dt.timedelta(0),
        'done': False
    })

    if run_stage in ('set', 'iter'):
        info['num_epochs_ran'] = 0
    
    if run_stage == 'set':
        info.update({
            'iter_info': get_run_stage_info('iter'),
            'epoch_info': get_run_stage_info('epoch'),

            'run_indices': np.arange(info['num_examples']),
            'score_arrs': {
                score_name: np.zeros(info['num_examples'])
                    for score_name in ('loss', 'rewards_sl', 'rewards_hsl')
            },
            'benchmarks': get_benchmarks(info['dataset'].get_glob('hsl_price_shift')),
        })
        
        if info['kind'] == 'train':
            info['run_indices'] = np.random.permutation(info['run_indices'])
        
        info['dataset'].writeable(True)

    return info

def get_benchmarks(hsl_price_shift):
    
    longe = hsl_price_shift[:, 2]
    short = hsl_price_shift[:, 1]

    # print('sum(longe)', sum(longe))
    # print('sum(longe > 0)', sum(longe > 0))
    
    assert sum(longe) > 0 or sum(short) > 0

    if sum(longe) > sum(short):
        main_type = 'long'
        main = longe
    elif sum(short) > sum(longe):
        main_type = 'short'
        main = short

    benchmarks = {
        'main_type': main_type,
        'main': main,
        'max': np.amax(hsl_price_shift[:], axis=1),
    }

    return benchmarks

def run_sets(agent, set_infos):

    results = {}
    for i, set_info in enumerate(set_infos):
        set_info['start'] = dt.datetime.now()
        info = Info(set_info, set_info['iter_info'], set_info['epoch_info'])
        print()
        info.print_stage_stats('set')
        
        if set_info['kind'] == 'train':
            agent.train()

        elif set_info['kind'] in ('val', 'test'):
            agent.eval()
        
        
        agent, info = run_iter(agent, info)
        info.set['time_ran'], info.set['start'] = add_time(info.set)
        info.print_stage_stats('set')
        
        set_results = get_results(info)
        for key, value in set_results.items():
            print(key, value)
        
        results.update(set_results)
        set_infos[i] = info.set

    return results, agent, set_infos

def add_time(an_info):
    time_ran = an_info['time_ran'] + (dt.datetime.now() - an_info['start'])
    new_start = dt.datetime.now()
    return time_ran, new_start

def get_results(info):
    
    results = {}
    for result_type, value in info.set['results'].items():
        if result_type == 'scores':
            
            for score_info in value:
                benchmarks = None if score_info['vs'] is None else info.set['benchmarks'][score_info['vs']]

                results[score_info['name']] = get_standardized_score(info.set['score_arrs'][score_info['raw']],
                                        benchmarks=benchmarks, indices=info.set['run_indices'], end=info.examples_ran('set'))

        elif result_type == 'epochs':
            if value:
                results['epochs'] = info.epochs_ran('set')
    
    # add the set name to the result name
    results = {f"{result_name}_{info.set['base_name']}": result_value for result_name, result_value in results.items()}

    return results

def get_standardized_score(raw_score, start=0, end=None, indices=None, benchmarks=None):
    # score is normalized by the number of examples in each batch

        score_arr = raw_score[start:end]
        stand_score = sum(score_arr)

        if benchmarks is not None:

            if indices is None:
                indices = np.arange(len(score_arr))
            else:
                indices = indices[start:end]

            stand_score = stand_score / sum(benchmarks[indices])
        
        else:
            stand_score = stand_score / len(score_arr)

        # if trunc is not None: stand_score = utils.trunc(stand_score, trunc)
        return stand_score

def info_inner_to_outer(outer_info, inner_info):
    outer_info['num_epochs_ran'] += inner_info['num_epochs_ran']
    outer_info['num_batches_ran'] += inner_info['num_batches_ran']
    outer_info['num_examples_ran'] += inner_info['num_examples_ran']

    inner_info.update(get_run_stage_info(inner_info['run_stage']))
    return outer_info, inner_info

def run_iter(agent, info):
    info.iter['start'] = dt.datetime.now()
    
    while True:

        agent, info = run_epoch(agent, info)

        info.iter['time_ran'], info.iter['start'] = add_time(info.iter)
        info.print_stage_stats('iter')
        
        if info.iter['done'] == True:
            info.iter.update(get_run_stage_info('iter'))
            return agent, info

def run_epoch(agent, info):
    info.epoch['start'] = dt.datetime.now()

    dataloader = get_loader_for_epoch(info.set, info.epoch)
    with torch.set_grad_enabled(info.set['kind'] == 'train'):
        for experiences, possible_rewards in dataloader:

            scores = agent.act_and_learn(experiences, possible_rewards, info.set['kind'], info.batches_ran('iter'))
            info.set['score_arrs'] = get_scores(info.set['score_arrs'], info.epoch['num_examples_ran'], scores, len(possible_rewards))

            for level in info.levels:
                getattr(info, level)['num_batches_ran'] += 1
                getattr(info, level)['num_examples_ran'] += len(possible_rewards)
            
            info.epoch['time_ran'], info.epoch['start'] = add_time(info.epoch)
            if info.epoch['num_examples_ran'] == info.set['num_examples']:
                
                info.iter['num_epochs_ran'] += 1
                info.set['num_epochs_ran'] += 1

                info.epoch.update(get_run_stage_info('epoch'))
                info.epoch['done'] = True
            
                if info.set['kind'] == 'train':
                    info.set['run_indices'] = np.random.permutation(info.set['run_indices'])
            
            
            info = periodic_check(info.set['periodics'], info)

            if info.iter['done'] or info.epoch['done']:
                info.epoch['done'] = False
                return agent, info

def get_loader_for_epoch(set_info, epoch_info):
    indices = set_info['run_indices'][epoch_info['num_examples_ran']:]
    subset = Subset(set_info['dataset'], indices)
    dataloader = DataLoader(subset, shuffle=False, batch_size=set_info['batch_size'], pin_memory=True,
                                                            num_workers=multiprocessing.cpu_count()-1)
    return dataloader

def get_scores(set_info_scores, epoch_examples_ran, scores, len_batch):
    for score_name, score in scores.items():
        set_info_scores[score_name ][epoch_examples_ran:epoch_examples_ran+len_batch] = \
                                        score.item() if score_name == 'loss' else score.numpy()

    return set_info_scores

def periodic_check(periodics, info):
    for periodic in periodics:
        for whm in periodic['what_how_many']:
            metric = getattr(info, periodic['metric'])(whm['level'])
            if metric >= whm['at_how_many']:
                if whm['do_what'] == 'stop':
                    if whm['level'] in ('set', 'iter'):
                        info.iter['done'] = True

    return info


# class Runner(object):
#     def __init__(self, hist):
#         self.hist = hist
    
#     # whole history
#     def get_all_examples_trained(self):
#         all_examples_trained = 0
#         for h in self.hist:
#             for ph, phas in h[' s'].items():
#                 if ph[0:5] == 'train':
#                     all_examples_trained += len(phas['indices'])*phas['epochs']+len(phas['rewards'])
        
#         return all_examples_trained

#     # run
#     def run(self, key=None):
#         if key == None:
#             return self.hist[-1]
#         else:
#             return self.hist[-1][key]

    
#     def phases_to_run(self):
        
#         iters = [ph['iter'] for ph in self.phases().values()]
#         if np.all(np.array(iters) == iters[0]):
#             return [phase for phase in self.phases().keys()]
#         else:
#             return [k for k, v in self.phases().items() if v['iter'] < max(iters)]
    
#     def phases(self):
#         return self.run('phases')

#     # phase
#     def pha(self, key=None):
#         if key == None:
#             return self.run('phases')[self.phase]
#         else:
#             return self.run('phases')[self.phase][key]
    
#     # epoch
    
        
#     def batches_ep(self):
#         be = self.pha('ep_index') // self.pha('batch_size')
#         if self.pha('ep_index') % self.pha('batch_size') != 0:
#             be += 1
#         return be
    
#     def batches_per_phase(self):
#         bpp = len(self.pha('indices')) // self.pha('batch_size')
#         if len(self.pha('indices')) % self.pha('batch_size') != 0:
#             bpp += 1
#         return bpp

#     def batches_run(self):
#         batches_run = self.pha('epochs') * self.batches_per_phase() + self.batches_ep()
#         return batches_run
    
#     def total_phase_epochs(self):
#         return self.pha('epochs') + self.pha('ep_index') / self.pha('m')
    



# you can visualize your model on tensorboard
# def visualize_model():
#     # v = cfg.v
#     writer = SummaryWriter()
#     writer.add_graph(v.model, v.trainset.X)
#     writer.close()
# writer.add_scalar('accuracy', epoch_acc, model.train_count)
# writer.add_scalars('epoch loss', {f'{phase}_loss': epoch_loss}, model.train_count)
# writer.add_scalars('epoch loss', {f'{phase}_loss': epoch_loss