import os, json, copy, ray, datetime as dt, numpy as np, torch
from ray import tune

from train import init, set_stuff, run
from utils import resources
from utils.misc import Objdict
import config, munge


def dump_json_args(all_args):
    json_args = copy.deepcopy(all_args.__dict__)
    for key, obj in json_args.items():
        json_args[key] = obj.__dict__
    
    json_args = json.dumps(json_args, default=lambda o: '<not serializable>', indent=4)
    with open('all_data_info.json', 'w') as json_data_info:
        json.dump(json_args, json_data_info, indent=4)

def load_args_from_json(checkpoint_dir):

    path = os.path.join(checkpoint_dir, 'all_data_info.json')
    with open(path, 'r') as json_data_info:
        args = json.load(json_data_info)
    
    return args

def init_run(args, cfg):
    print('The dependent variables:\n', args)

    # Needed for both run and munge
    args = config.Args(args, cfg)
    dump_json_args(args)

    # Specific to munge.
    munger = munge.main.Munger(args.munge, inspect=cfg.inspect)
    set_infos = munger.munge()

    # Specific to run.
    if cfg.val_only:
        for i, set_info in enumerate(set_infos):
            if set_info['kind'] == 'train':
                del set_infos[i]
    
    agent = init.Agent(args.run)
    set_infos = [run.get_run_stage_info('set', info=set_info) for set_info in set_infos]

    return set_infos, agent
    
def empty_init(self):
    pass

def solo_run(config, stop, go_run=True):
    Trainer.__init__ = empty_init
    trainer = Trainer()
    trainer._setup(config, go_solo=True)

    if go_run:
        stop_key, stop_value = list(stop.items())[0]

        while True:    
            results = trainer._train()
            trainer._save('solo_val_dir')
            print('stop_key, stop_value:', stop_key, stop_value)
            print('results[stop_key]', results[stop_key])
            if results[stop_key] == stop_value:
                return results
    
    else: return

class Trainer(tune.Trainable):    
    def _setup(self, varbs_config, go_solo=False):

        print('_setup varbs_config', varbs_config)
        print('go_solo', go_solo)
        
        self.run_outer_cfg = config.Outer(go_solo)
        self.set_infos, self.agent = init_run(varbs_config, self.run_outer_cfg)
        # I can resume a run by saving whatever I need to continue in beliefs package.
        # when beliefs load, so too will whatever history I saved. I can designate in run args
        # I can add a function to update set_infos with what is held in loaded beliefs.
        
    def _train(self):
        results, self.agent, self.set_infos = run.run_sets(self.agent, self.set_infos)
        
        # custom checkpoint
        if 'sl_vs_max_test' in results:
            self.agent.beliefs_package['score'] = results['sl_vs_max_test']
            init.reflect(self.agent.beliefs, self.agent.beliefs_package, 'test_checkpoints')

        return results

    def _save(self, checkpoint_dir):
        self.agent.beliefs_package['history'][-1]['set_infos'] = [
            {key: value for key, value in set_info.items() if key != 'dataset'}
                for set_info in self.set_infos
        ]
        init.write_beliefs(self.agent.beliefs, self.agent.beliefs_package, checkpoint_dir)
        return checkpoint_dir

    def _restore(self, checkpoint_dir):
        self.agent.beliefs, self.agent.beliefs_package = init.read_beliefs(checkpoint_dir,
                                                                        self.agent.beliefs.device,
                                                                        self.agent.beliefs)
        self.set_infos = self.agent.history[-1]['set_infos']
        args = load_args_from_json(checkpoint_dir)
        
        munger = munge.main.Munger(args, inspect=True)
        set_infos = munger.munge()
        
        for i, set_info in enumerate(set_infos):
            self.set_infos[i]['dataset'] = set_info['dataset']
        
    def reset_config(self, varbs_config):
        del self.set_infos
        self.set_infos, self.agent, = init_run(varbs_config, self.run_outer_cfg)
        return True # I don't know why, but return true

def go():
    varbs_cfg = config.Varbs()
    if varbs_cfg.go_tune:
    
        cfg = config.Tune()

        # "Restart Ray defensively in case the ray connection is lost."
        ray.shutdown()
        ray.init(log_to_driver=False)
        
        if cfg.test:
            cfg.kwargs['name'] = 'test'
            cfg.rm_results_dir = True
            cfg.kwargs['num_samples'] = 1
            cfg.kwargs['fail_fast'] = True

        if cfg.rm_results_dir:
            experiment_path = os.path.join(cfg.kwargs['local_dir'], cfg.kwargs['name'])
            os.system(f"rm -rf {experiment_path}")

        cfg.kwargs['config'] = varbs_cfg.kwargs
        results = tune.run(Trainer, **cfg.kwargs)
        print("Best config:\n", results.get_best_config(metric=cfg.kwargs['checkpoint_score_attr'], mode='max'))
        print("Best logdir:\n", results.get_best_logdir(cfg.kwargs['checkpoint_score_attr'], mode="max")) 
    
    else:

        cfg = config.Solo()
        cfg.kwargs['config'] = varbs_cfg.kwargs
        results = solo_run(**cfg.kwargs)

    return results