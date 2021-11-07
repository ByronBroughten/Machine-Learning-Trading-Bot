#%% import and initialize
import os, sys, copy, zarr, torch, datetime as dt, numpy as np, pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, Subset
from IPython.display import display

import utils, munge
pack_runner = utils.misc.Pack_Runner(globals())

class Munger(object):

    # maybe reconsider how you're naming everything. 'Split' might be the other stage, alongside glob, not dataset.
    # dataset might be reserved for the end, when the splits are put into the pytorch dataset format.

    # maybe adjust how file-creation is done and unify the variables of file_creaton, make, and inspect

    def __init__(self, args, inspect=True, training=True):
        
        munge.paths.del_tmps(args.outer_path)

        self.outer_path = args.outer_path
        self.load_in_memory = args.load_in_memory
        
        for i, glob_info in enumerate(args.glob_infos):
            for j, stage_info in enumerate(glob_info['datastages']):
                args.glob_infos[i]['datastages'][j]['file'] = utils.misc.string_from_keylists_dic(args.__dict__, stage_info['file'])
        self.glob_infos = args.glob_infos

        data_outer, self.file_key = munge.paths.get_path_pieces(args.outer_path, args.data_path, args.price_data_info)

        self.varbs = {
            'general': {
                'dtype': args.np_dtype,
                'seed': args.seed,
                'outer_path': args.outer_path,
                'seq_len': args.seq_len,
                'horizon': args.horizon,
                'data_outer': data_outer
            }
        }
        
        if training: # gotta figure out how this might differ for live--hopefully not much

            set_batches, self.num_total_final = get_munge_set_batch_info(args.set_batches, args.seed)
            self.full_glob_trimmer = munge.process.Full_Glob_Trimmer(args.seq_len, self.num_total_final)
            
            self.file_key, raw_path = munge.paths.init_training_munge(data_outer, self.file_key)
            basic_path = munge.paths.get_datapath(data_outer, 'basic', '', 'basic', self.file_key, args.basic_stage['file_type'])
            
            self.varbs['general'].update({
                'pack_runner': pack_runner,
                'varbs': self.varbs,
                'num_total_final': self.num_total_final,
                'file_key': self.file_key,
                'basic_path': basic_path,
            })

            self.total_basic = munge.basic.get_total_basic(args.seq_len, self.num_total_final, args.horizon,
                                                            args.basic_stage['access'], self.varbs, raw_path)
            self.basic_glob = munge.process.get_basic(self.total_basic, 'basic', args.seq_len, self.num_total_final, 0)
            self.basic_glob = self.full_glob_trimmer.trim(self.basic_glob)
            
            self.set_infos = get_set_infos(set_batches, self.basic_glob['datetime'].to_numpy())

    def munge(self):
        print("It's mungin' time.")

        final_set_infos = []
        for set_info in self.set_infos:
            self.varbs['dataset'] = get_dataset_varbs(set_info)
            # for reference outside of munge.

            for glob_info in self.glob_infos:
                self.varbs['glob'] = get_glob_varbs(glob_info, self.num_total_final)
                self.full_glob_trimmer.glob_kind = glob_info['kind']
                if self.set_infos[0]['kind'] == 'train' and glob_info['X_or_Y'] == 'X':
                    self.varbs['glob']['scaler_path'] = munge.paths.get_scaler_path(self.outer_path, glob_info['name'], 
                                                                                    self.file_key, self.set_infos[0]['name'])
                
                dataset = self.total_basic

                to_make = []
                for stage in reversed(glob_info['datastages']):
                    self.varbs['stage'] = get_stage_varbs(stage, set_info['name'], glob_info['name'], self.varbs)

                    data, done, self.varbs = init_stage(stage, self.varbs, self.full_glob_trimmer, self.varbs[stage['kind']]['num_examples'])
                    
                    # the num_examples needed for glob... is it the total num needed or the post-trim needed?
                            
                    if done:
                        dataset = data
                        if 'inspect' in stage:
                            for pack in stage['inspect']:
                                pack_runner(pack, self.varbs, dataset)
                        break
                    elif not done:
                        to_make = [stage] + to_make
                
                for stage in to_make:

                    self.varbs['stage'] = get_stage_varbs(stage, set_info['name'], glob_info['name'], self.varbs)
                    dataset = run_stage(dataset, stage, self.varbs)
                    if stage['kind'] == 'glob':
                        dataset = self.full_glob_trimmer.trim(dataset)

        
                glob_info['dataset'] = dataset
                set_info['glob_infos'][glob_info['name']] = glob_info
            
            set_info['dataset'] = munge.set_stuff.get_XY_dataset(set_info['glob_infos'])
            print(f"Got dataset for {set_info['name']}")
            final_set_infos.append(set_info)
            
        if self.load_in_memory:
            final_set_infos = munge.set_stuff.load_seqs_as_possible(final_set_infos, exclude='train')
        
        for set_info in final_set_infos:
            set_info['dataset'].writeable(False)
            
        # final inspection
        for glob_info in self.glob_infos:
            print('\nGlob name:', glob_info['name'])

            test_glob = pd.DataFrame()
            # self.full_glob_trimmer.glob_kind = glob_info['kind']
            self.varbs['glob'] = get_glob_varbs(glob_info, self.num_total_final)
            self.varbs['glob']['min_basic_idx'] = self.basic_glob.index.min()

            for set_info in final_set_infos:
                print('\nSet name:', set_info['name'])

                self.varbs['dataset'] = get_dataset_varbs(set_info)
                self.varbs['dataset']['dataset'] = set_info['dataset'].get_glob(glob_info['name'])


                basic_dataset = dataset_test(self.basic_glob, glob_info['inspect'], self.varbs)

                if glob_info['inspect']['test_dataset_funcs']:
                    print(f"{set_info['name']} {glob_info['name']} passed the set-span test.")

                if glob_info['inspect']['get_basic_cols_funcs']:
                    test_glob=test_glob.append(basic_dataset, ignore_index=False, verify_integrity=True, sort=True)
            
            test_glob = test_glob.sort_index()
            
            # if glob_info['name'] == 'hsl_price_shift': # hsl_price_shift
            #     print("self.basic_glob\n", self.basic_glob[90000:90030])
            #     print("test_glob\n", test_glob[90000:90030])

            print(f"\nStarting the {glob_info['name']} glob test.")
            for func_pack in glob_info['inspect']['test_glob_funcs']:
                pack_runner(func_pack, self.varbs, test_glob)
            print(f"{glob_info['name']} glob passed the glob test.")

        print('All dataset globs passed the tests: Yay!')
        munge.paths.del_tmps(self.outer_path)

        return final_set_infos

def dataset_test(basic_glob, inspect, varbs):

    basic_dataset = basic_glob
    for func_pack in inspect['get_basic_cols_funcs']:
        basic_dataset = pack_runner(func_pack, varbs, basic_dataset)

    for func_pack in inspect['test_dataset_funcs']:
        pack_runner(func_pack, varbs, basic_dataset)

    return basic_dataset

def run_stage(dataset, stage, varbs):
    print(f"\n{stage['name']} stage.")

    pack_containers = [] # I could put the pack containers in a thing in stage
    for container_name in ('process', 'inspect', 'access'):
        if container_name in stage: pack_containers.append(stage[container_name])
    try_gather_varbs(pack_runner, (pack_containers), varbs)
        
    if isinstance(dataset, zarr.core.Array):# Make a temporary zarr array that isn't finalized until save_or_append.
        dataset = utils.zarrs.get_copy(dataset, f"{varbs['general']['outer_path']}/tmp/{stage['name']}")
    
    for pack in stage['process']:
        dataset = pack_runner(pack, varbs, dataset)
    
    for pack in stage['inspect']:
        pack_runner(pack, varbs, dataset)
    
    # puts the zarr array in the permanent path.
    dataset = pack_runner(stage['access']['save_or_append'], varbs, dataset)

    return dataset

def init_stage(stage, varbs, full_glob_trimmer, num_needed):

    # as things stand, num_total_previous is the length of the data pre-trim.i
    # num_needed is the length of the data post-trim. num_needed is just len(basic_glob)
    
    data = pack_runner(stage['access']['load'], varbs)

    if stage['kind'] == 'glob':
        varbs['glob']['num_total_previous'] = len(data)
        print('num_total_previous', len(data))
        data = full_glob_trimmer.trim(data)
    
    if len(data) < num_needed:
        done = False
    elif len(data) == num_needed:
        done = True
    elif len(data) > num_needed and stage['kind'] == 'glob':
        done = True
    elif len(data) > num_needed and stage['kind'] == 'dataset':
        print(f"There's data for the {stage['kind']} stage, but its length is {len(data)}",
            f"rather than {num_needed}.", f"Here, see the data: \n{data}")
        raise ValueError('See print.')

    if full_glob_trimmer.glob_kind == 'price_shift' and stage['kind'] == 'glob':
        print('len price_shift', len(data))
        print('num_needed', num_needed)
        print('done', done)
        # sys.exit()
    
    return data, done, varbs

def try_gather_varbs(pack_runner, func_pack_bundles, all_varbs):
    for bundle in func_pack_bundles:
        if isinstance(bundle, dict):
            bundle = bundle.values()
            for func_pack in bundle:
                pack_runner(func_pack, all_varbs, dry_run=True)

    print('All varbs and modules seem to be in order.')

def get_set_infos(set_batches, timestamps):

    set_infos = []

    for sb in set_batches:
        for set_info in sb['set_infos']:
            name = set_info['base_name']
            
            if sb['shuffle']:
                
                num = ''
                for si in sb['set_infos']:
                    num = num + '-' + str(si['num_examples'])
            else:
                num = '-' + str(set_info['num_examples'])

            name += num
            name += '-' + str(timestamps[sb['index']])

            set_info.update({
                'shuffle': sb['shuffle'],
                'index': sb['index'],
                'num_sb_examples': sb['num_sb_examples'],
                'indices': sb['indices'][set_info['base_name']],
                'name': name,
                'is_train': set_info['kind'] == 'train',
                'glob_infos': {}
            })

            set_infos.append(set_info)
    
    return set_infos

def get_munge_set_batch_info(set_batches, seed):
      
    examples_thus_far = 0
    for i, sb in enumerate(set_batches):
        
        sb['index'] = examples_thus_far # This is before num_sb_examples added to it.

        sb['num_sb_examples'] = 0
        for set_info in sb['set_infos']:
            sb['num_sb_examples'] += set_info['num_examples']
        
        t_range = torch.tensor([i for i in range(sb['index'], sb['index'] + sb['num_sb_examples'])])
        sb['indices'] = munge.split.split_main(t_range, sb['set_infos'], sb['shuffle'], seed)

        for key, indices in sb['indices'].items():
            indices = indices.tolist()
            indices.sort()
            sb['indices'][key] = indices

        examples_thus_far += sb['num_sb_examples']
        set_batches[i] = sb

    return set_batches, examples_thus_far

def count_data(data, num_needed, stage_kind):
    if len(data) < num_needed:
        done = False
    elif len(data) == num_needed:
        done = True
    elif len(data) > num_needed and stage_kind == 'glob':
        done = True
    elif len(data) > num_needed and stage_kind == 'dataset':
        print(f"There's data for the {stage_kind} stage, but its length is {len(data)}",
            f"rather than {num_needed}.", f"Here, see the data: \n{data}")
        raise ValueError('See print.')
    return done

# def count_data(data, num_needed, stage_kind):
#     if len(data) == 0:
#         done = False; valid = False
    
#     elif len(data) == num_needed or (stage_kind == 'glob' and len(data) > num_needed):
#         done = True; valid = True
    
#     elif len(data) < num_needed and stage_kind == 'glob':
#         done = False; valid = True
    
#     elif stage_kind == 'dataset' and len(data) != num_needed:
#         print(f"There's data for the {stage_kind} stage, but its length is {len(data)}",
#             f"rather than {num_needed}.", f"Here, see the data: \n{data}")
#         raise ValueError('See print.')            
        
#     return done, valid

def get_dataset_varbs(set_info):
    varbs = {key: value for key, value in set_info.items() if key != 'dataset'}
    return varbs
    
def get_glob_varbs(glob_info, num_examples):

    varbs = {
        'name': glob_info['name'],
        'kind': glob_info['kind'],
        'num_examples': num_examples
    }
    
    if 'varbs' in glob_info:
        varbs.update(glob_info['varbs'])
    
    return varbs

def get_stage_varbs(stage_info, set_name, glob_name, all_varbs):

    varbs = {}

    if 'varbs' in stage_info:
        varbs.update(stage_info['varbs'])
    
    path_pack = {
        'func': 'munge.paths.get_datapath',
        'varbs': {
            'kwargs': {
                'general': ['data_outer', 'file_key'],
                stage_info['file_type_varb_location']: ['file_type']
            }
        },
        'kwargs': {
            'data_stage': stage_info['kind'],
            'mid_path': os.path.join(set_name, glob_name) if stage_info['kind'] == 'dataset' else glob_name,
            'file_name': stage_info['file'],
        }
    }
    
    all_varbs['stage'] = varbs
    
    varbs.update({
        'name': stage_info['name'],
        'kind': stage_info['kind'],
        'path': pack_runner(path_pack, all_varbs)
    })
    
    return varbs

# For checking things out.
# import munge
# basic = munge.access.load_hdf5('/mnt/teradactyle/big_bbtrade/data/training/chart_equity_subs_mins1/basic/basic-IBM_kbots_adj_1998.h5')
# context_glob = munge.access.load_hdf5('/mnt/teradactyle/big_bbtrade/data/training/chart_equity_subs_mins1/glob/context/1-weekday-hour-minute-IBM_kbots_adj_1998.h5')