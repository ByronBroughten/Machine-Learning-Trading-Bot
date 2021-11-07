import os, sys, multiprocessing, json, math, numpy as np
from ray import tune

import utils

class Live(object):
    def __init__(self):
        
        train_first = False
        train_kwargs = {} # it can maybe get train kwargs from config, but also not if you'd prefer.
        
        # Outer data path that differentiates its data source from training. This will have to interact with path.
        # config_path, for a configuration for its model
        # it can also have a model path. if it doesn't have the model path, it will use the configuration path to make the model,
        # and also 
        config_path = ''

class Varbs(object):
    def __init__(self, seq_len=None, horizon=None):
        outer_path = os.path.dirname(os.getcwd())
        print('outer_path:', outer_path)

        
        self.go_tune = True
        self.load_config = False
        self.load_model = False # always comes with config
        self.model_paths = {
            'config_path': "/mnt/big_bubba/ray_results_store/phase_one/Trainer_1_batch_size=4096,clip=0.77606,dropout=0.24768,horizon=13,kernel=9,seq_len=90,levels=26,loss_func=MSELoss,tcn_channel_size_2020-08-29_10-58-20hnk5canf/params.json",
            'model_path': "/mnt/big_bubba/ray_results_store/phase_one/Trainer_1_batch_size=4096,clip=0.77606,dropout=0.24768,horizon=13,kernel=9,seq_len=90,levels=26,loss_func=MSELoss,tcn_channel_size_2020-08-29_10-58-20hnk5canf/test_checkpoints/11.757480734195328|0/beliefs.pth",
        }

        if self.load_model:
            with open(self.model_paths['config_path']) as json_config: 
                self.kwargs = json.load(json_config)
            self.kwargs['model_path'] = self.model_paths['model_path']

        else: # one week: 1950
            seq_len_tune = tune.choice([60, 90, 180, 390, 780, 1170]) if seq_len is None else tune.choice([seq_len])
            seq_len_solo = 780 if seq_len is None else seq_len

            horizon_tune = tune.choice([i for i in range(1, 25)]) if horizon is None else tune.choice([horizon])
            horizon_solo = 8 if horizon is None else horizon

            kernel = tune.choice([i for i in range(2, 10)]) if self.go_tune else 9

            self.kwargs = {
                # 'seq_k_and_l': tune.choice(get_seqs_ks_and_ls(seq_lens, kernels)) if go_tune else\
                #                             else {'seq_len': 1536, 'kernel': 10, 'levels': 11,},
                'seq_len': seq_len_tune if self.go_tune else seq_len_solo,
                'horizon': horizon_tune if self.go_tune else horizon_solo,
                'kernel': kernel,
                'levels': tune.sample_from(lambda spec: np.random.choice(get_seqs_ks_and_ls(
                                spec.config.seq_len, spec.config.kernel)))\
                            if self.go_tune else 12,#max(get_seqs_ks_and_ls(seq_len_solo, kernel)),
                'chans': tune.sample_from(lambda spec: int(np.random.choice(get_possible_channels(spec.config.seq_len)))) if self.go_tune else 16,
                'drop': tune.uniform(.05, .5) if self.go_tune else 0.27297,
                'clip': tune.uniform(0.3, 1) if self.go_tune else 0.95165,
                'loss_func':  tune.choice(['SmoothL1Loss', 'MSELoss']) if self.go_tune else 'SmoothL1Loss',
                'b_size': tune.sample_from(lambda spec: int(np.random.choice(
                                            batch_size_for_seq(spec.config.seq_len))))\
                                        if self.go_tune else 1024, #1024, 2048, 4096
                
                'outer_path': outer_path
            }

class Outer(object):
    def __init__(self, go_solo):
        
        # if not go_solo, then the args are for a tune run.
        self.inspect = True if go_solo else True
        self.quick_test = False if go_solo else False
        self.val_only = False if go_solo else False

# time_stop = .1 if quick_test else 3 # 5 .05
epoch_stop_run = 7
class Solo(object):
    def __init__(self):
        self.kwargs = {
            'stop': {'epochs_train': epoch_stop_run},
            'go_run': True,
        }

        print(f"go_run is set to {self.kwargs['go_run']}")

class Tune(object):
    def __init__(self):
        self.test = False
        self.rm_results_dir = False
        self.kwargs = {
            'name': 'phase_four',
            'local_dir': '/mnt/big_bubba/ray_results_store',

            # search_alg: bayes
            # scheduler: tune.schedulers.ASHAScheduler(metric="score")
            
            'num_samples': 100,
            'stop': {'epochs_train': epoch_stop_run},

            'max_failures': 0,
            'fail_fast': False,

            'checkpoint_score_attr': 'max_vs_hsl_val',
            'checkpoint_freq': 1,
            'checkpoint_at_end': True,
            'keep_checkpoints_num': 2,
            'global_checkpoint_period': 300,

            'reuse_actors': True,
            'resources_per_trial': {
                "cpu": multiprocessing.cpu_count()-1,
                "gpu": 1
            },
        }

class Munge(object):
    def __init__(self, varbs, cfg):
        # super(Munge, self).__init__()

        self.data_type_str = 'float32'
        self.np_dtype = utils.misc.get_dtype_np_torch(self.data_type_str, 'np')

        self.seed = 1111
        self.load_in_memory = True

        example_stop = 10000 if cfg.quick_test else 250000
        # for zarr array operations
        at_once_max = 100000
        at_once_half = at_once_max // 2
        at_once_fifth = at_once_max // 5

        # I could standardize this
        if 'seq_len' in varbs: self.seq_len = varbs['seq_len']
        if 'horizon' in varbs: self.horizon = varbs['horizon']
        if 'seq_k_and_l' in varbs: self.seq_k_and_l = varbs['seq_k_and_l']
        
        # self.data_label = 'IBM_adj_1998_min1'
        self.outer_path = varbs['outer_path'] #'/home/buster/big_bbtrade'
        self.data_path = 'data/training'
        self.big_outer = '/mnt/big_bubba/'
        
        # context_hot_fit = ['minute'] + ['hour' + str(i) for i in range (7)] + ['weekday' + str(i) for i in range (5)]
        self.step_mins = varbs['step_mins']

        self.basic_stage = {
            'name': 'basic',
            'file_type': 'h5',
            'access': {
                'load': {
                    'func': 'munge.access.try_load',
                    'varbs': {'args': [{'general': ['basic_path']}]},
                    'kwargs': {
                        'file_type': 'h5',
                        'file_where': 'in_memory'
                    }
                },
                'save_or_append': {
                    'func': 'munge.access.append',
                    'varbs': {'args': [{'general': ['basic_path', 'num_basic_previous']}]},
                    'kwargs': {
                        'file_type': 'h5',
                        'file_where': 'in_memory',
                    }
                }
            },
            'file': 'basic'
        }

        self.glob_infos = [
            {
                'name': 'seqs', # this could be called something related to the model that will be used.
                'kind': 'seqs',
                'X_or_Y': 'X',
                'varbs': {
                    'file_type': 'zarr',
                    'file_where': 'on_disk',
                    'seq_cols_info': [
                        {
                            'made_from': {
                                'basic_column': 'close',
                                'seq_idxs': {
                                    'channel': 3,
                                    'interval': self.seq_len-1
                                }
                            },
                            'standardized_with': {
                                'basic_column': 'open',
                                'seq_idxs': {
                                    'channel': 0,
                                    'interval': 0
                                }
                            }
                        },{
                            'made_from': {
                                'basic_column': 'volume',
                                'seq_idxs': {
                                    'channel': 4,
                                    'interval': 3 # np.random.choice(range(self.seq_len))
                                }
                            },
                            'standardized_with': None
                        }    
                    ]
                },
                'datastages': [
                    {
                        # how am I going to do the packages? Either I insert them here (but then they aren't JSON serealizable) or I 
                        # pass a string reference to them (but then I have to use pairs)
                        'name': 'glob',
                        'kind': 'glob',
                        'file_type_varb_location': 'glob',
                        'access': {
                            'load': {
                                'func': 'munge.access.try_load',
                                'varbs': {
                                    'kwargs': {
                                        'glob': ['file_type', 'file_where'],
                                        'stage': ['path']
                                    }   
                                },
                            },
                            'save_or_append': {
                                'func': 'munge.access.append',
                                'varbs': {
                                    'kwargs': {
                                        'glob': ['file_type', 'file_where', 'num_total_previous'],
                                        'stage': ['path']
                                    }
                                },
                            },
                        },
                        'process': [# seqs
                            {
                                'func': 'munge.process.get_basic',
                                'varbs': {
                                    'args': [{'glob': ['kind']}],
                                    'kwargs': {
                                        'general': ['num_total_final'],
                                        'glob': ['num_total_previous']
                                    }
                                },
                                'kwargs': {'seq_len': self.seq_len,}
                            },{
                                'func': 'munge.process.make_seqs',
                                'kwargs': {
                                    'seq_len': self.seq_len,
                                    'features': ['open', 'high', 'low', 'close', 'volume'],
                                    'zarr_args': {
                                        'at_once': at_once_max,
                                        'arr_kwargs': get_zarr_kwargs(chunk=int(at_once_fifth), clevel=7, dtype=self.np_dtype),
                                        'outer_path': self.outer_path
                                    }
                                },
                            },
                        ],
                        'inspect': [],
                        'file': [
                            ['seq_len'],
                            ['step_mins'],
                            ['glob_infos', 0, 'datastages', 1, 'process', 2, 'kwargs', 'scalers', 0, [['to_fit']]]
                        ]
                    },{
                        'name': 'final',
                        'kind': 'dataset',
                        'file_type_varb_location': 'glob',
                        'varbs': {
                            'zarr_args': {
                                'at_once': at_once_max,
                                'arr_kwargs': get_zarr_kwargs(chunk=at_once_fifth, clevel=7, dtype=self.np_dtype),
                                'outer_path': self.outer_path
                            }
                        },
                        'access': {
                            'load': {
                                'func': 'munge.access.try_load',
                                'varbs': {
                                    'kwargs': {
                                        'glob': ['file_type', 'file_where'],
                                        'stage': ['path']
                                    }
                                }
                            },
                            'save_or_append': {
                                'func': 'munge.access.save',
                                'varbs': {
                                    'kwargs': {
                                        'glob': ['file_type', 'file_where'],
                                        'stage': ['path']
                                    }
                                }
                            }
                        },
                        'process': [
                            {
                                'func': 'munge.process.split',
                                'varbs': {
                                    'kwargs': {
                                        'dataset': ['indices'],
                                        'stage': ['zarr_args']
                                    },
                                },
                            },{
                                'func': 'munge.process.stand_each_seq',
                                'varbs':{'kwargs': {'stage': ['zarr_args']}},
                            },{ 
                                'func': 'munge.process.scale',
                                'varbs': {
                                    'kwargs': {
                                        'glob': ['scaler_path'],
                                        'dataset': ['is_train']
                                    }
                                },
                                'kwargs': {
                                    'd_format': 'seqs',
                                    'scalers': [
                                        {
                                            # 'path': "../scalers/IBM_adj_1998_min1-train-883733400-1460000-300000/RobustScaler-basic-open",
                                            'package': 'sklearn',
                                            'obj_name': 'RobustScaler',
                                            'seq_len': self.seq_len,
                                            'to_fit': ['open', 'high', 'low', 'close'],
                                            'to_transform': 'same_as_fit',
                                        },{
                                            # 'path': "../scalers/IBM_adj_1998_min1-train-883733400-1460000-300000/RobustScaler-basic-open",
                                            'package': 'sklearn',
                                            'obj_name': 'RobustScaler',
                                            'seq_len': self.seq_len,
                                            'to_fit': ['volume'],
                                            'to_transform': 'same_as_fit',
                                        }
                                    ],
                                    'zarr_args': {
                                        'at_once': at_once_max
                                    }
                                }
                            },{
                                'func': 'munge.process.change_zarr_params',
                                'kwargs': {
                                    'at_once': at_once_max,
                                    'kwargs': {
                                        'chunk': 1,
                                        'clevel': 5,
                                    },
                                    'outer_path': self.outer_path
                                }
                            },{
                                'func': 'munge.process.switch_arr_modes',
                                'kwargs': {'mode': 'r'}
                            }
                        ],
                        'inspect': [
                            {
                                'func': 'munge.inspect.run_simple_test_packs',
                                'varbs': {
                                    'kwargs': {
                                        'general': ['varbs', 'pack_runner'],
                                        'glob': ['file_type', 'file_where'],
                                    },
                                },
                                'kwargs': {'fast_packs_only': True,}
                            },
                        ],
                        'file': [
                            ['seq_len'],
                            ['step_mins'],
                            ['glob_infos', 0, 'datastages', 1, 'process', 2, 'kwargs', 'scalers', 0, [['package'], ['obj_name'], ['to_fit'], ['to_transform']]],
                            ['glob_infos', 0, 'datastages', 1, 'process', 2, 'kwargs', 'scalers', 1, [['obj_name'], ['to_fit'], ['to_transform']]]
                        ]
                    }
                ],
                'inspect': {
                    'get_basic_cols_funcs': [
                        # {
                        #     'func': 'munge.process.split',
                        #     'varbs': {'kwargs': {'dataset': ['indices']}},
                        # },{
                        #     'func': 'munge.inspect.get_cols_arr_seqs_3d',
                        #     'varbs': {
                        #         'args': [
                        #             {'dataset': ['dataset']},
                        #             {'glob': ['name']}
                        #         ],
                        #         'kwargs': {'glob': ['seq_cols_info']}
                        #     },        
                        # }
                    ],
                    'test_dataset_funcs': [
                        # {
                        #     'func': 'munge.inspect.set_span_test',
                        #     'varbs': {
                        #         'args': [{'dataset': ['index']}],
                        #         'kwargs': {
                        #             'glob': ['min_basic_idx'],
                        #             'dataset': ['num_sb_examples', 'shuffle']
                        #         }
                        #     }
                        # }
                    ],
                    'test_glob_funcs': [
                        # {
                        #     'func': 'munge.inspect.test_seqs_3d',
                        #     'varbs': {
                        #         'args': [{'glob': ['name']}],
                        #         'kwargs': {
                        #             'general': ['seq_len'],
                        #             'glob': ['seq_cols_info'],
                        #         }
                        #     }
                        # }
                    ]
                }
            },{
                'name': 'context',
                'kind': 'context',
                'X_or_Y': 'X',
                'varbs': {
                    'col_num_names': [
                        {'num': 0,
                        'name': 'minute'}
                    ]
                },
                'datastages': [
                    {
                        'name': 'glob',
                        'kind': 'glob',
                        'file_type_varb_location': 'stage',
                        'varbs': {
                            'file_type': 'h5',
                            'file_where': 'in_memory',
                        },
                        'access': {
                            'load': {
                                'func': 'munge.access.try_load',
                                'varbs': {
                                    'kwargs': {
                                        'stage': ['file_type', 'file_where', 'path']
                                    }
                                },
                            },
                            'save_or_append': {
                                'func': 'munge.access.append',
                                'varbs': {
                                    'kwargs': {
                                        'glob': ['num_total_previous'],
                                        'stage': ['file_type', 'file_where', 'path']
                                    }
                                },
                            }
                        },
                        'process': [
                            {
                                'func': 'munge.process.get_basic',
                                'varbs': {
                                    'args': [{'glob': ['kind']}],
                                    'kwargs': {
                                        'general': ['num_total_final'],
                                        'glob': ['num_total_previous']
                                    }
                                },
                                'kwargs': {'seq_len': self.seq_len}
                            },{
                                'func': 'munge.process.make_context',
                                'kwargs': {
                                    'features': [
                                        {
                                            'onehot': ['weekday', 'hour'],
                                            'not': ['minute'],
                                        }
                                    ],
                                },
                            },
                        ],
                        'inspect': [],
                        'file': [
                            ['step_mins'],
                            ['glob_infos', 1, 'datastages', 0, 'process', 1, 'kwargs', 'features', 0, [['onehot'], ['not']]]
                        ] 
                    },{
                        'name': 'final',
                        'kind': 'dataset',
                        'file_type_varb_location': 'stage',
                        'varbs': {
                            'file_type': 'zarr',
                            'file_where': 'in_memory'
                        },
                        'access': {
                            'load': {
                                'func': 'munge.access.try_load',
                                'varbs': {'kwargs': {'stage': ['file_type', 'file_where', 'path']}}
                            },
                            'save_or_append': {
                                'func': 'munge.access.save',
                                'varbs': { 'kwargs': {'stage': ['file_type', 'file_where', 'path']}},
                            }
                        },
                        'process': [
                            {
                                'func': 'munge.process.split',
                                'varbs': {'kwargs': {'dataset': ['indices']}},
                            },{
                                'func': 'munge.process.scale',
                                'varbs': {
                                    'kwargs': {
                                        'glob': ['scaler_path'],
                                        'dataset': ['is_train']
                                    }
                                },
                                'kwargs': {
                                    'd_format': 'context',
                                    'scalers': [
                                        {
                                            # 'path': "../scalers/IBM_adj_1998_min1-train-883733400-1460000-300000/RobustScaler-context-minute-weekday0-weekday1-weekday2-weekday3-weekday4-hour0-hour1-hour2-hour3-hour4-hour5-hour6",'
                                            'package': 'sklearn',
                                            'obj_name': 'RobustScaler',
                                            'to_fit': 'all_as_is',
                                            'to_transform': 'same_as_fit',
                                        }
                                    ]
                                }
                            },{
                                'func': 'munge.process.np_change_dtype',
                                'kwargs': {'dtype': self.np_dtype}
                            }
                        ],
                        'inspect': [
                            {
                                'func': 'munge.inspect.run_simple_test_packs',
                                'varbs': {
                                    'kwargs': {
                                        'general': ['varbs', 'pack_runner'],
                                        'stage': ['file_type', 'file_where']
                                    }
                                },
                            }
                        ],
                        'file': [
                            ['seq_len'],
                            ['step_mins'],
                            ['glob_infos', 1, 'datastages', 0, 'process', 1, 'kwargs', 'features', 0, [['onehot'], ['not']]],
                            ['glob_infos', 1, 'datastages', 1, 'process', 1, 'kwargs', 'scalers', 0, [['package'], ['obj_name'], ['to_fit'], ['to_transform']]]
                        ]
                    },
                ],
                'inspect': {
                    'get_basic_cols_funcs': [
                        {
                            'func': 'munge.process.split',
                            'varbs': {'kwargs': {'dataset': ['indices']}},
                        },{
                            'func': 'munge.inspect.get_cols_arr_2d',
                            'varbs': {
                                'args': [
                                    {'dataset': ['dataset']},
                                    {'glob': ['name', 'col_num_names']}
                                ]
                            }
                        }
                    ],
                    'test_dataset_funcs': [
                        {
                            'func': 'munge.inspect.set_span_test',
                            'varbs': {
                                'args': [{'dataset': ['index']}],
                                'kwargs': {
                                    'glob': ['min_basic_idx'],
                                    'dataset': ['num_sb_examples', 'shuffle']
                                }
                            }
                        }
                    ],
                    'test_glob_funcs': [
                        {
                            'func': 'munge.inspect.cols_relative_basic',
                            'varbs': {'args': [{'glob': ['name', 'col_num_names']}]}
                        }
                    ]
                }
            },{
                'name': 'hsl_price_shift',
                'kind': 'price_shift',
                'X_or_Y': 'Y',
                'varbs': {
                    'col_num_names': [
                        {'num': 0,
                        'name': 'hold'},
                        {'num': 1,
                        'name': 'short'},
                        {'num': 2,
                        'name': 'long'}
                    ],
                    'now_col_name': 'close',
                    'prev_col_name': 'close'
                },
                'datastages': [
                    {
                        'name': 'glob',
                        'kind': 'glob',
                        'file_type_varb_location': 'stage',
                        'varbs': {
                            'file_type': 'h5',
                            'file_where': 'in_memory'
                        },
                        'access': {
                            'load': {
                                'func': 'munge.access.try_load',
                                'varbs': {
                                    'kwargs': {
                                        'stage': ['file_type', 'file_where', 'path']
                                    }
                                },
                            },
                            'save_or_append': {
                                'func': 'munge.access.append',
                                'varbs': {
                                    'kwargs': {
                                        'glob': ['num_total_previous'],
                                        'stage': ['file_type', 'file_where', 'path']
                                    }
                                }
                            }
                        },
                        'process': [
                            {
                                'func': 'munge.process.get_basic',
                                'varbs': {
                                    'args': [{'glob': ['kind']}],
                                    'kwargs': {
                                        'general': ['num_total_final'],
                                        'glob': ['num_total_previous']
                                    }
                                },
                                'kwargs': {'seq_len': self.seq_len,}
                            },{
                                'func': 'munge.process.price_shift_buy_sell_hold',
                                'kwargs': {
                                    'features': ['hold', 'short', 'long'],
                                    'horizon': self.horizon,
                                }
                            },
                        ],
                        'inspect': [],
                        'file': [
                            ['step_mins'],
                            ['glob_infos', 2, 'datastages', 0, 'process', 1, 'kwargs', [['horizon'], ['features']]]
                        ]
                    },{
                        'name': 'final',
                        'kind': 'dataset',
                        'file_type_varb_location': 'stage',
                        'varbs': {
                            'file_type': 'zarr',
                            'file_where': 'in_memory'
                        },
                        'access': {
                            'load': {
                                'func': 'munge.access.try_load',
                                'varbs': {'kwargs': {'stage': ['file_type', 'file_where', 'path']}}
                            },
                            'save_or_append': {
                                'func': 'munge.access.save',
                                'varbs': {'kwargs': {'stage': ['file_type', 'file_where', 'path']}}
                            }
                        },
                        'process': [
                            {
                                'func': 'munge.process.split',
                                'varbs': {'kwargs': {'dataset': ['indices']}},
                            },{
                                'func': 'munge.process.df_to_numpy',
                                'kwargs': {'dtype': self.np_dtype}
                            },
                        ],
                        'inspect': [
                            {
                                'func': 'munge.inspect.run_simple_test_packs',
                                'varbs': {
                                    'kwargs': {
                                        'general': ['varbs', 'pack_runner'],
                                        'stage': ['file_type', 'file_where']
                                    }
                                }
                            }
                        ],
                        'file': [
                            ['seq_len'],
                            ['step_mins'],
                            ['glob_infos', 2, 'datastages', 0, 'process', 1, 'kwargs', [['horizon'], ['features']]]
                            
                        ]
                    }
                ],
                'inspect': {
                    'get_basic_cols_funcs': [
                        {
                            'func': 'munge.process.split',
                            'varbs': {'kwargs': {'dataset': ['indices']}},
                        },{
                            'func': 'munge.inspect.get_cols_arr_2d',
                            'varbs': {
                                'args': [
                                    {'dataset': ['dataset']},
                                    {'glob': ['name', 'col_num_names']}
                                ]
                            }
                        }
                    ],
                    'test_dataset_funcs': [
                        {
                            'func': 'munge.inspect.set_span_test',
                            'varbs': {
                                'args': [{'dataset': ['index']}],
                                'kwargs': {
                                    'glob': ['min_basic_idx'],
                                    'dataset': ['num_sb_examples', 'shuffle']
                                }
                            }
                        }
                    ],
                    'test_glob_funcs': [
                        {
                            'func': 'munge.inspect.sl_ys_to_close',
                            'varbs': {
                                'args': [{'glob': ['name']}],
                                'kwargs': {
                                    'glob': ['col_num_names', 'now_col_name', 'prev_col_name'],
                                    'general': ['horizon']
                                }
                            }
                        }
                    ]
                }
            }
        ]
        self.price_data_info = {
            # chart_equity_subs--minute ohlcv candlesticks
            # nasdaq_book_subs--I think a stream of nasdaq trades or something
            'training_and_trading': {
                'tda_datatype': 'chart_equity_subs',
                'step_mins': varbs['step_mins'],
            },
            'training_only': {
                'symbol': 'IBM',
                'vendor': 'kbots',
            },
        }
        self.all_data_info = {
            'price_data_info': self.price_data_info,
            'glob_infos': self.glob_infos,
        }

        # 1460000, 300000, 500000 (200000)
        self.set_batches = [
            {
                'shuffle': True,
                'set_infos': [
                    
                    {
                        'base_name': 'train',
                        'kind': 'train',
                        'num_examples': 20000 if cfg.quick_test else 1460000,# 1460000,
                        
                        'batch_size': varbs['b_size'],
                        'periodics': [
                            {
                                'metric': 'examples_ran',
                                'what_how_many': [
                                    {'do_what': 'stop',
                                    'level': 'iter',
                                    'at_how_many': example_stop}
                                ]
                            },{
                                'metric': 'epochs_ran',
                                'what_how_many': [
                                    {'do_what': 'stop',
                                    'level': 'set',
                                    'at_how_many': epoch_stop_run}
                                ]
                            }
                        ],
                        'results': {
                            'scores': [
                                {
                                    'raw': 'rewards_sl',
                                    'vs': 'max',
                                    'name': 'max_vs_hsl'
                                },{
                                    'raw': 'rewards_hsl',
                                    'vs': 'main',
                                    'name': 'main_vs_hsl'
                                },{
                                    'raw': 'loss',
                                    'vs': None,
                                    'name': 'loss'
                                }
                            ],
                            'epochs': True
                        }
                    },{
                        'base_name': 'val',
                        'kind': 'val',
                        'num_examples': 15000 if cfg.quick_test else 300000, # 300000,
                        'batch_size': varbs['b_size'] * 2,
                        'periodics': [
                            {
                                'metric': 'epochs_ran',
                                'what_how_many': [
                                    {'do_what': 'stop',
                                    'level': 'iter',
                                    'at_how_many': 1}
                                ]
                            }
                        ],
                        'results': {
                            'scores': [
                                {
                                    'raw': 'rewards_sl',
                                    'vs': 'max',
                                    'name': 'max_vs_hsl'
                                },{
                                    'raw': 'rewards_sl',
                                    'vs': 'main',
                                    'name': 'main_vs_sl'
                                },{
                                    'raw': 'rewards_hsl',
                                    'vs': 'main',
                                    'name': 'main_vs_hsl'
                                },{
                                    'raw': 'loss',
                                    'vs': None,
                                    'name': 'loss'
                                }
                            ]
                        },
                    }
                ],
            },{
                'shuffle': False,
                'set_infos': [
                    {
                        'base_name': 'test',
                        'kind': 'test',
                        'num_examples': 12000 if cfg.quick_test else 200000, # 200000,
                        'batch_size': varbs['b_size'] * 3,
                        'periodics': [
                            {
                                'metric': 'epochs_ran',
                                'what_how_many': [
                                    {'do_what': 'stop',
                                    'level': 'iter',
                                    'at_how_many': 1}
                                ]
                            }
                        ],
                        'results': {
                            'scores': [
                                {
                                    'raw': 'rewards_sl',
                                    'vs': 'max',
                                    'name': 'max_vs_hsl'
                                },{
                                    'raw': 'rewards_hsl',
                                    'vs': 'main',
                                    'name': 'main_vs_hsl'
                                },{
                                    'raw': 'rewards_sl',
                                    'vs': 'main',
                                    'name': 'main_vs_sl'
                                },{
                                    'raw': 'loss',
                                    'vs': None,
                                    'name': 'loss'
                                }
                            ]
                        }
                    },
                ],
            }
        ]

        # making folder to identify data
        self.independent_varbs = ['data_label']
        
class Run(object):
    def __init__(self, munge, varbs):
        # if 'model_path' in varbs:
        #     self.model_params = {'path': varbs['model_path']}

        self.new_run = True # can only be false if loading model is true.
        self.torch_dtype = utils.misc.get_dtype_np_torch(munge.data_type_str, 'torch')
        self.model_params = {
            'make': 'TCN_FC_1D',
            'dtype': self.torch_dtype,
            'model_kwargs': {
                'TCN_kwargs': {
                    'seq_len': munge.seq_len,
                    'input_size': 5, # sum([len(i['final']) for i in munge.glob_infos['seqs']['features']]),
                    'dropout': varbs['drop'],
                    'output_size': varbs['tcn_output_size'],
                    'channel_size': varbs['chans']
                },
                'context_kwargs': { # I could get these later after munge
                    'input_size': 13, # sum([len(i['final']) for i in munge.glob_infos['seqs']['features']])
                    'out_size_arr': varbs['context_out_size_arr']
                },
                'output': 2, # len(munge.glob_infos['ys']['features']) - 1
            },
        }
        if 'model_path' in varbs:
            self.model_params['path'] = varbs['model_path']
        
        else:
            mk = self.model_params['model_kwargs']
            
            if 'kernel' in varbs: mk['TCN_kwargs']['kernel'] = varbs['kernel']
            if 'levels' in varbs: mk['TCN_kwargs']['levels'] = varbs['levels']
            input_size =  mk['TCN_kwargs']['output_size'] + mk['context_kwargs']['out_size_arr'][-1]

            mk['final_kwargs'] = {
                'input_size': input_size,
                'out_size_arr': [input_size]
            }

        # run parameters
        # Adam lr 2e-3?
        self.clip = varbs['clip']

        self.loss_func = varbs['loss_func'] # SmoothL1Loss, MSELoss, BCELoss, CrossEntropyLoss
        self.batch_sizes = {
            'train': varbs['b_size'],
            'val': varbs['b_size'] * 2,
            'test': varbs['b_size'] * 2
        }
        
        # stop the run when both of these are the case?
        # self.stop_change = .00001
        # self.stop_time = 24
        self.num_actions = 3 #len(munge.glob_infos['ys']['features'])

        # these are what are kept track of. they define the folder and model names
        self.model_name_varbs = [
            'loss_func',
            'clip',
            ['batch_sizes', ['train']]
        ]

# put the rest of init here?
class Args(object):
    def __init__(self, varbs, cfg):
        # super(Args, self).__init__()
        
        possible_varbs = {
            'context_mode': 'onehot',
            'tcn_output_size': 10,
            'context_out_size_arr': [26, 20, 13, 10],
            'step_mins': 1, # tune.choice([l for l in range(1, 390 + 1) if 390 % l == 0 or l == 390])
        }
        varbs.update(possible_varbs)
        self.munge = Munge(varbs, cfg)
        self.run = Run(self.munge, varbs)

        # if cfg.go_run:
        # else:
        #     self.run = None
        # self = varbs_post_tune_choice(self)


class Client(object):
    def __init__(self):
        # get these here: https://developer.tdameritrade.com/user/me/apps
        self.api_key = 'FFDXWIK6WPVIAV6KUATEHGZ0LNJNQLGG' # also known as "consumer key"
        self.redirect_url = 'https://localhost' # also known as "callback url"
        # this is where you want requests to be sent

        # download chrome web_driver here: https://selenium-python.readthedocs.io/installation.html#drivers
        # cd Downloads & unzip chromedriver_linux64.zip
        # sudo mv chromedriver /usr/bin/chromedriver
        # sudo chown root:root /usr/bin/chromedriver
        # sudo chmod +x /usr/bin/chromedriver
        self.webdriver_path = '/home/buster/big_bbtrade/tb_run/tda/auth_stuff/chromedriver'
        self.token_path = '/home/buster/big_bbtrade/tb_run/tda/auth_stuff/token.pkl'
        self.account_id = '497141542'
        
        # trial dir will contain the relevant belief and feature configs that were used when moulding the beliefs
        
class Trader(object):
    def __init__(self):
        
        self.acceptable_lag_secs = 10

        self.symbol = 'IBM'

        self.trial_dir = 'placeholder_trial_dir'
        self.beliefs_dir = 'placeholder_beliefs_dir'

class Tda(object):
    def __init__(self):
        self.clients = Client()
        self.handler = Trader()


######################################################################################

def get_zarr_kwargs(dtype=None, chunk=None, clevel=None, order=None):
# this is like a mini config itself of default zarr values
    if dtype == 'float':
        dtype = 'f4'

    zarr_kwargs = {
        'dtype': 'f4' if dtype is None else dtype,
        'chunk': 1 if chunk is None else chunk,
        'clevel': 5 if clevel is None else clevel,
        'order': 'C' if order is None else order,
    }
    return zarr_kwargs


def update(dic1, dic2):
    dic1.update(dic2)
    return dic1

def get_levels(k, seq_len):
# returns an array of all possible numbers of levels given sequence length, kernel, and max extra.
    max_extra = 28

    levels_nums = []
    
    for i in range(1000):
        d = 2**i

        rec_field = k + d * (k - 1)
        two_kernel_range = 2 + d * (2 - 1)
        
        if rec_field >= seq_len / 2:
            if two_kernel_range <= seq_len:
                levels_nums.append(i + 1)
            else: break
    
    last_level = levels_nums[-1]

    # 64 is the smallest seq_len, and 1280 is the largest that will fit on GPU without extra
    # 1280 // 64 = 20
    extra = max_extra - seq_len // 64
    
    for i in range(1, extra + 1):
        levels_nums.append(last_level + i)
    
    return levels_nums

def get_kernels_and_levels(kernels, seq_len):
# if I really wanted I could make this a list comprehension
# max_extra_levels would end up being an array, and everything in range(100) loop
# would be in a function            
    
    ks_and_ls = []
    for k in kernels:
        
        levels_nums = get_levels(k, seq_len)
        
        # plus one because we want at least one number (0) for each kernel
        for l in levels_nums:
            ks_and_ls.append({'kernel': k, 'levels': l})
    
    return ks_and_ls

def get_seqs_ks_and_ls(seq_lens, kernels):
    
    if isinstance(seq_lens, list) and isinstance(kernels, list):
        seqs_ks_and_ls = [update(k_l, {'seq_len': ls}) for ls in seq_lens for\
                            k_l in get_kernels_and_levels(kernels, ls)]
    elif isinstance(seq_lens, list):
        seqs_ks_and_ls = [k_l for k_l in get_kernels_and_levels(kernels, seq_lens)]
    
    # neither are lists. just get levels.
    else:
        seqs_ks_and_ls = get_levels(kernels, seq_lens)
    
    return seqs_ks_and_ls

def batch_size_for_seq(seq_len):
    batch_sizes = [1024, 2048, 4096]

    if seq_len > 700: return batch_sizes[:1]
    # elif seq_len > 500: return batch_sizes[:2]
    else: return batch_sizes[:2]

def get_possible_channels(seq_len):
    if seq_len > 700:
        maxi = 25
        max_seq = 1170 - 700
        seq_len = seq_len - 700
    else:
        maxi = 40
        max_seq = 700 - 30
        seq_len = seq_len - 30

    mini = 8 # how low mini can actually be
    lowest_max = 12 # this is the lowest the max can be
    diff = maxi - lowest_max

    how_many_extra = math.floor((1 - (seq_len/max_seq)) * diff)
    if how_many_extra < 0: how_many_extra = 0

    total = how_many_extra + lowest_max

    
    channels = [i for i in range(mini, total+1)]

    return channels

def varbs_post_tune_choice(args):
    if hasattr(args.munge, 'seq_k_and_l'):
        args.munge['seq_len'] = args.munge['seq_k_and_l']['seq_len']
        
        if args.run:

            tcn_kwargs = args.run.model_params['model_kwargs']['TCN_kwargs']
            tcn_kwargs['kernel'] = args.munge.seq_k_and_l['kernel']
            tcn_kwargs['levels'] = args.munge.seq_k_and_l['levels']


        del args.munge.seq_k_and_l

    return args

# for live and simulated-live, to be expanded upon in those
        # 'tv': {
        #     'train': {
        #         'size': 5
        #     },
        #     'val': {
        #         'size': 5
        #     }
        # },
        # # this is for any zarr pics that are periodically created and overwritten during live run
        # 'zarr_kwargs': { #
        #     'dtype': 'f2',
        #     'chunk_size': 1,
        #     'clevel': 5,
        #     'order': 'C',
        # },
        # 'learn': {
        #     'every': 5,
        #     'for': 10
        # },
        # train_every
        # 390 = 1 trading day
        # 1950 = 5 trading days


        # supervise-specific
        #'val_batch_size': 64,
        #'pred_func': 'sigmoid',

        # reinforcement-specific
        # 'mem_size': sets_meta['y']['rewards']['dim'][0],
        # eps = {
        #     'n': 0,
        #     'decay': 0.99,
        #     'min': 0.01,
        #     'dec': .2
        # }
        # 'replace': 1000,

# #%%
# import pandas as pd
# basic = pd.read_hdf(f"../glob/basic/{space['munge']['data_label']}")
# maximum_examples = len(basic) - 207 - 1 # minus 207 for making Xs. minus 1 for making ys
# sample_size = utils.get_sample_size(me=.005, ci=.999)
# # .005 margin of error and .999 confidence interval seems what image net deemed appropriate
# # .01 and .99 works ok for 50000 and up

# print(maximum_examples - sample_size * 6, sample_size, sample_size * 5



# schedulers = {}
# schedulers['pbt'] = PopulationBasedTraining(
#     time_attr="training_iteration",
#     metric="mean_accuracy",
#     mode="max",
#     perturbation_interval=5,
#     hyperparam_mutations={
#         # distribution for resampling
#         "lr": lambda: np.random.uniform(0.0001, 1),
#         # allow perturbations within this set of categorical values
#         "momentum": [0.8, 0.9, 0.99],
#     }
# )
# u_kwargs = {
#         'kind': 'ucb',
#         'kappa': 2.576,
#         'xi': 0.0
#         },    
# search_alg = tune.suggest.bayesopt.BayesOptSearch(args, metric="score", mode="max",
#                                                             utility_kwargs=u_kwargs)