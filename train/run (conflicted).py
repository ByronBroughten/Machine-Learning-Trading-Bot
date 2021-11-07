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
        self.set = set_info
        self.iter = iter_info
        self.epoch = epoch_info
    
    def examples_ran(self, level):
        if level in ('set', 'iter'):
            num_examples = getattr(self, level)['num_examples_ran'] + self.epoch['num_examples_ran']
        elif level == 'epoch':
            num_examples = self.epoch['num_examples_ran']
        else: raise ValueError("Received level variable not in ('set', 'iter', 'epoch')")
        return num_examples
    
    def batches_ran(self, level):
        if level in ('set', 'iter'):
            num_batches = getattr(self, level)['num_batches_ran'] + self.epoch['num_batches_ran']
        elif level == 'epoch':
            num_batches = self.epoch['num_batches_ran']
        return num_batches
    
    def epochs_ran(self, level):
        if level in ('set', 'iter'):
            num_epochs = getattr(self, level)['num_epochs_ran'] + self.epochs_ran('epoch')
        elif level == 'epoch':
            num_epochs = self.epoch['num_examples_ran'] / self.set['num_examples']
        return num_epochs

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
        
        if set_info['kind'] == 'train':
            agent.train()

        elif set_info['kind'] in ('val', 'test'):
            agent.eval()
        
        info = Info(set_info, set_info['iter_info'], set_info['epoch_info'])
        agent, info = run_iter(agent, info)
        
        info.set, info.iter = info_inner_to_outer(info.set, info.iter)
        info.set['time_ran'], info.set['start'] = add_time(info.set)
        
        results.update(get_results(info))
        set_infos[i] = info.set

    return results, agent, set_infos

def add_time(an_info):
    time_ran = an_info['time_ran'] + (dt.datetime.now() - an_info['start'])
    new_start = dt.datetime.now()
    return time_ran, new_start

def get_results(info):
    
    results = {}
    for result_type, value in info.set['results']:
        if result_type == 'scores':
            
            for score_info in value:
                benchmarks = None if score_info['vs'] is None else info.set['benchmarks'][score_info['vs']]

                results[score_info['name']] = get_standardized_score(info.set['score_arrs'][score_info['raw']],
                                        benchmarks=benchmarks, indices=info.set['run_indices'], end=info.examples_ran('set'))

        elif result_type == 'epochs':
            if result_type['epochs']:
                results['epochs'] = info.epochs_ran('set')
    
    # add the set name to the result name
    results = {f"{result_name}_{info.set['name']}": result_value for result_name, result_value in results.items()}

    return results

def get_standardized_score(raw_score, start=0, end=None, indices=None, benchmarks=None):
    # score is normalized by the number of examples in each batch

        stand_score = raw_score[start:end]
        stand_score = sum(stand_score)

        if benchmarks is not None:

            if indices is None:
                indices = np.arange(len(stand_score))
            else:
                indices = indices[start:end]

            score = score / sum(benchmarks[indices])
        
        else:
            stand_score = stand_score / len(stand_score)

        # if trunc is not None: stand_score = utils.trunc(stand_score, trunc)
        return stand_score

def info_inner_to_outer(outer_info, inner_info):
    outer_info['num_batches_ran'] += inner_info['num_batches_ran']
    outer_info['num_examples_ran'] += inner_info['num_examples_ran']

    if ['num_epochs_ran'] in inner_info:
        outer_info['num_epochs_ran'] += inner_info['num_epochs_ran']

    inner_info.update(get_run_stage_info(inner_info['run_stage']))
    return outer_info, inner_info

def run_iter(agent, info):

    info.iter['start'] = dt.datetime.now()
    
    while True:
        agent, info = run_epoch(agent, info)
        info.iter['time_ran'], info.iter['start'] = add_time(info.iter)
        
        if info.iter['done'] == True:
            return agent, info

def run_epoch(agent, info):
    info.epoch['start'] = dt.datetime.now()

    dataloader = get_loader_for_epoch(info.set, info.epoch)
    with torch.set_grad_enabled(info.set['kind'] == 'train'):
        for experiences, possible_rewards in dataloader:

            scores = agent.act_and_learn(experiences, possible_rewards, info.set['kind'], info.batches_ran('iter'))
            info.set['score_arrs'] = get_scores(info.set['score_arrs'], info.epoch['num_examples_ran'], scores, len(possible_rewards))

            info.epoch['num_batches_ran'] += 1
            info.epoch['num_examples_ran'] += len(possible_rewards)
            info.epoch['time_ran'], info.epoch['start'] = add_time(info.epoch)

            
            if info.epoch['num_examples_ran'] == info.set['num_examples']:
                info.iter['num_epochs_ran'] += 1
                info.iter, info.epoch = info_inner_to_outer(info.iter, info.epoch)
            
                if info.set['kind'] == 'train':
                    info.set['run_indices'] = np.random.permutation(info.set['run_indices'])
                
                info.epoch['done'] = True
            
            
            info.iter['done'] = periodic_check(info.set['periodics'], info)

            if info.iter['done'] or info.epoch['done']:
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
            if whm['at_how_many'] >= metric:
                if whm['do_what'] == 'stop':
                    getattr(info, whm['level'])['done'] = True

    return info

def print_phase_stats(runner, result):
    print(runner.phase)
    print(f"epochs {runner.total_phase_epochs()}")
    print(dt.datetime.now() - runner.pha('start'))
    print('examples', runner.pha('examples'))
    for k, v in result.items():
        print(k, v)
    print()

def show_train_stats(run, pha):
    print(f"total run time {dt.datetime.now() - run.start['total']}, "
        f"epoch {utils.trunc(pha.batches_run()/pha.batches_per_phase(), 3)}, "
        f"train batches {run.batches_trained()} ")

def report(run, pha):
    
    loss = pha.score('loss', start=-run.batches['report'], trunc=6)
    score = pha.score('reward', start=-run.batches['report'], trunc=6)
    print(f"last {run.batches['report']} batches, loss {loss}, score {score} "
            f"epoch {utils.trunc(pha.batches_run()/pha.batches_per_phase(), 3)}")
    # show_train_stats(run, pha)

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
# writer.add_scalars('epoch loss', {f'{phase}_loss': epoch_loss}, d