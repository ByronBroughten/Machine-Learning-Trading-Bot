#%%
import torch, pickle, os, shutil
import datetime as dt
import numpy as np

import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from collections import namedtuple

from train import custom_models, set_stuff
from utils.misc import Objdict
import utils


#%% Reinforcement Learning

# epsilon or no epsilon
    # eps_params = ('n', 'decay', 'min', 'dec')
    # try: self.eps = args.eps
    # except AttributeError:
    #     self.eps = Objdict()
    #     for p in eps_params: self.eps[p] = 0
        
    # anything to do with model params


event_pieces = ('experience', 'action', 'reward')
class Agent(object):
    def __init__(self, args):
        print('Cuda Available:', torch.cuda.is_available())
        
        self.seq_len = args.model_params['model_kwargs']['TCN_kwargs']['seq_len']
        self.action_args = [i for i in range(args.num_actions)]

        # get model params
        device = utils.pt.check_device()
        if 'path' in args.model_params:
            print(f"loading model from {args.model_params['path']}")
            self.beliefs, self.beliefs_package = read_beliefs(args.model_params['path'], device)
        else:
            self.beliefs, self.beliefs_package = custom_models.get(args, **args.model_params)
            print('Loading beliefs onto device.')
            self.beliefs.device = device
            self.beliefs = self.beliefs.to(self.beliefs.device)
        
        if args.new_run:
            self.beliefs_package['history'].append({})
        elif len(self.beliefs_package['history']) == 0:
            raise ValueError("Trying to resume an old run but this run history is empty.")

        # the rest of it
        self.loss_func = getattr(torch.nn, args.loss_func)()
        self.batch_sizes = args.batch_sizes
        self.clip = args.clip
        self.num_batches_ran = 0

        # I could make it check whether train is present.
        print('Getting grad scaler.')
        self.scaler = torch.cuda.amp.GradScaler()
        

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    # def decrement_epsilon(self):
    #     if self.eps.n > self.eps.min: self.eps.n = self.eps.n - self.eps.dec
    #     else: self.eps.n = self.eps.min

    def eval(self):
        # self.save_epsilon = self.eps.n
        # self.eps.n = 0
        self.beliefs.eval()
    
    def train(self):
        self.beliefs.train()
        # try: self.eps.n == self.save_epsilon
        # except AttributeError: pass


    def act_and_learn(self, experience, rewards, phase_type, num_batches_ran):
        
        try: # experience may be a dict
            for kind, qualia in experience.items():
                experience[kind] = qualia.to(self.beliefs.device) # torch.tensor(v).float().to(self.beliefs.device)
        except AttributeError: # or else it's just a tensor
            experience = experience.to(self.beliefs.device)

        # forward pass
        if phase_type == 'train':
            self.beliefs.optimizer.zero_grad()

        # make a prediction and observe
        q_actual = rewards[:, 1:].to(self.beliefs.device) # torch.tensor(rewards[:, 1:]).float().to(self.beliefs.device)
        
        # mixed precisions
        with torch.cuda.amp.autocast(enabled=False): #(phase_type=='train')
            q_pred = self.beliefs.forward(experience)
            loss = self.loss_func(q_actual, q_pred) #.to(self.beliefs.device)

        if phase_type == 'train': 
            
            self.scaler.scale(loss).backward()

            # for gradient clipping
            self.scaler.unscale_(self.beliefs.optimizer)

            if self.clip > 0:
                torch.nn.utils.clip_grad_norm_(self.beliefs.parameters(), self.clip)

            self.scaler.step(self.beliefs.optimizer)
            # self.beliefs.optimizer.step()
            self.scaler.update()
        
        # just short or long
        sl_action = (torch.argmax(q_pred, 1) + 1).tolist()

        # hold short or long
        hold = torch.zeros((len(q_pred), 1), device=self.beliefs.device)
        full_q_pred = torch.cat((hold, q_pred), 1)
        hsl_action = torch.argmax(full_q_pred, 1).tolist()

        scores = {
            'rewards_hsl': rewards[range(len(rewards)), hsl_action],
            'rewards_sl': rewards[range(len(rewards)), sl_action],
            'loss': loss
        }

        self.num_batches_ran += 1
        # if self.num_batches_ran % 50 == 0:
        #     # print('experience', experience)
        #     print('q_pred[:5]', q_pred[:3])
            # print('full_q_pred[:5]', full_q_pred[:5])
            # print('q_actual[:5]', q_actual[:5])
            # print("rewards['hsl'][:5]\n", rewards['hsl'][:5])
        return scores

# I can either copy this to my other class or I can make this into a function that spits out hist and beliefs
def read_beliefs(beliefs_dir, device=None, previous_beliefs=None):
    
    if device is None:
        device = utils.pt.check_device()

    # beliefs
    beliefs_dict = torch.load(beliefs_dir)
    if previous_beliefs is None or previous_beliefs.make != beliefs_dict['make']:
        beliefs = custom_models.get_custom_model(beliefs_dict['make'],
                                                    beliefs_dict['kwargs'])
    else:
        beliefs.to('cpu')
    beliefs.load_state_dict(beliefs_dict['beliefs'])
    beliefs.to(device)
    beliefs.device = device
    beliefs.name = beliefs_dict['name']

    # optimizer
    beliefs.optimizer = getattr(torch.optim, beliefs_dict['opt_make'])(beliefs.parameters())
    beliefs.optimizer.load_state_dict(beliefs_dict['optimizer'])

    for_beliefs_not_package = ('beliefs', 'optimizer', 'device')
    for bd in beliefs_dict:
        if bd in for_beliefs_not_package:
            del beliefs_dict[bd]

    print(f'The beliefs have been read from {beliefs_dir}.')
    return beliefs, beliefs_dict
    
def write_beliefs(beliefs, beliefs_package, dir_path):
    # payload to save
    beliefs.to(torch.device('cpu'))
    beliefs_dict = {
        'beliefs': beliefs.state_dict(),
        'optimizer': beliefs.optimizer.state_dict(),
    }
    beliefs.to(beliefs.device)
    beliefs_dict.update(beliefs_package)

    Path(dir_path).mkdir(parents=True, exist_ok=True)
    path = os.path.join(dir_path, "beliefs.pth")
    
    torch.save(beliefs_dict, path)
    print('wrote beliefs')

def get_dir_score(dir_name):
    score = float(dir_name.split('|')[0])
    return score

def reflect(beliefs, beliefs_package, folder, keep=1, minimize=False):
# write beliefs only if the score is better than what's there, and overwrite it
    score = beliefs_package['score']

    print("Reflecting.")
    # new_beliefs_best = True
    # relevant_files = self.get_beliefs_same_traits(same_name=False, mode='best')

    try:
        sub_folders = sorted([str(bl) for bl in os.listdir(folder)], key=get_dir_score)
    except FileNotFoundError:
        sub_folders = []
    
    how_many_better = 0
    new_checkpoint_num = 0
    for sf in sub_folders:

        if get_dir_score(sf) > score: how_many_better += 1
        checkpoint_num = float(sf.split('|')[1])
        if checkpoint_num > new_checkpoint_num:
            new_checkpoint_num = checkpoint_num + 1

    # if I want to keep 1, but zero were better, I write the new belief
    if keep > how_many_better:
        sub_folder = f'{score}|{new_checkpoint_num}'
        folder = os.path.join(folder, sub_folder)
        write_beliefs(beliefs, beliefs_package, folder)
        new = 1
    else:
        new = 0
        print("The score is not better than what we already have.")
        
    for _ in range((len(sub_folders) + new) - keep):
        shutil.rmtree(os.path.join(folder, sub_folders.pop(0)), ignore_errors=True)


# def write_event(self, event):

#     event_store = pickle.load(f'live/event_store/{self.data_label}.pk')
    
#     for p in event_store:
#         if p == 'experience':
#             for e in event_store[p]:
#                 event_store[p][e].append(event[p][e])
        
#         else: event_store[p].append(event[p])

#     pickle.dump(event_store, f'live/event_store/{self.data_label}.pk')


#%% Supervised Learning
# class Run(object):
#     # data_type, batch_size, sample_size, hidden layer size, hidden layers
#     # some means of regularization. eventually, different kinds of models.
#     def __init__(self, args):
#     # possible I could change this whole thing into a named tuple.
#     # I don't see any of the values needing to be edited with another
#     # named tuple inside of the named tuple. The only thing is that
#     # epochs trained couldn't be saved here. But that would work just as well being saved to the model.

#         # device
#         try:
#             device = torch.device(args['device'])
#         except KeyError:
#             device, args['device'] = utils.pt.get_device()


#         # get the model, account for variable number of parameters
#         model_param_names = ['model', 'data_type', 'device']
#         model_param_args = [args['model'], args['data_type'], device]

#         maybe_in_meta = ['pic_dim', 'fc_m', 'y_dim']
#         for i in maybe_in_meta:
#             try: model_param_args.append(args.sets_meta[i]), model_param_names.append(i)
#             except KeyError: pass
        
#         try: model_param_args.append(args['dense_layers']), model_param_names.append('dense_layers')
#         except KeyError: pass

#         Model_Params = namedtuple('Model_Params', model_param_names)
#         mp = Model_Params(*model_param_args)

        
#         # now, finally, get the model (and optimizer), with your variable number of parameters
#         self.model = custom_models.get(mp)
        
#         self.model = utils.pt.change_data_type(self.model, mp.data_type)
#         self.model.device = mp.device
#         self.model = self.model.to(self.model.device)
#         # not sure if I have to do this before assigning optimizer to model
#         # and not sure i want to change data_type of pretrained model

        
#         # functions for loss and prediction
#         self.loss_func = getattr(torch.nn, args['loss_func'])()
#         self.pred_func = utils.pt.get_pred_func(args['pred_func'])

        
#         # ready the sets and data loaders
#         sets = set_stuff.get_torch_sets(args['data'], args['data_type'])

#         batch_size = args['batch_size']
#         try: val_batch_size = args['val_batch_size']
#         except KeyError: val_batch_size = len(sets['val'].y)
        
#         if 'train' in sets and 'val' in sets:
#             print('sets working as intended, maybe')
#             self.dataloaders = utils.pt.get_loaders_train_and_val(
#                 sets['train'], sets['val'], batch_size, val_batch_size)
        
#     def get_args(self):
#         return [self.model, self.loss_func, self.pred_func, self.dataloaders]

    # def learn_from_actual(self, value, advantage, rewards):

    # # get action, not random or random
    # if np.random.random() > self.eps.n:
    #     action = torch.argmax(advantage, 1).tolist()
    #     # action = torch.argmax(advantage).item() # maybe if batch_size == 1
    
    # else:
    #     # action if experience is dict-like    
    #     try: action = [np.random.choice(self.action_args) for i\
    #                 in range(len(list(rewards.values())[0]))]
    #     # or if it's its own matrix
    #     except TypeError: action = [np.random.choice(self.action_args) for i\
    #                                                 in range(len(rewards))]
    # # print('rewards\n', rewards)
    # # print('action\n', action)

    # # I don't know why, but you have to use range(len()) instead of :
    # q_actual = rewards[range(len(rewards)), action]
    # q_actual = torch.tensor(q_actual).float().to(self.beliefs.device)
    # # add the values to the advantages to get the total predicted values
    # q_pred = torch.add(value, (advantage - advantage.mean(dim=1, keepdim=True)))
    # # get the predicted value of the action that was chosen
    # action = torch.tensor(action).long().to(self.beliefs.device)
    # q_pred = torch.gather(q_pred, 1, action.unsqueeze(-1)).squeeze(-1)

    # actual_reward = q_actual

    # return q_pred, q_actual, actual_reward



        # reading and writing
    # def get_beliefs_dir(self, mode=None):
    #     modes = ('best', 'checkpoint')
    #     if mode in modes: kind = mode
    #     else: raise FileNotFoundError(f'no mode in {modes} was entered')
        
    #     return f'beliefs/{kind}/{self.data_label}|{self.phase}'
    
    # def get_beliefs_path(self, mode=None):
    #     if mode == 'best': label = self.beliefs.score
    #     elif mode == 'checkpoint': label = self.get_all_examples_trained()
    #     return f'{self.get_beliefs_dir(mode)}/{self.beliefs.name}|{label}'

    # def get_beliefs_same_traits(self, same_name=False, mode=None):
    #     pieces = 3 if same_name else 2

    #     try:
    #         files = [str(bl) for bl in os.listdir(self.get_beliefs_dir(mode=mode)) if\
    #                                                     utils.misc.split_text_where(str(bl), '|', pieces)[0] == \
    #                                                     utils.misc.split_text_where(self.beliefs.name, '|', pieces)[0]]
    #     except FileNotFoundError:
    #         print(f'no directory found with the name {self.get_beliefs_dir(mode=mode)}')
    #         files = []
            
    #     return files