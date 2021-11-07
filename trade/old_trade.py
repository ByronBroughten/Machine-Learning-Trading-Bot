import pickle
import set_stuff
from munge import utils
from utils.misc import Objdict

class Online_TD_Env():
    def __init__(self):
        self.basic = None

        self.short_term_memory = pass # basic-like row
    
    def get_and_process_td_quote(self):
        raw = get_quote()
        
        basic_point = preprocess(raw)
        
        self.basic.append(basic_point)
        experience = get_experience(self.basic)

        return experience

        # preprocess will be a subset of

    def calc_reward(self):
        
        # this should access my account information and account for slippage
        present = self.basic['close'][-1]
        previous = self.basic['close'][-2]

        theoretical_rewards = get_rewards(present, previous)
        
        # we're goign to start with theoretical. we'll need tick-by-tick to account for
        # slippage when training and enmass
        # actual_rewards = get_rewards(self.price_at_trade, previous)
        # slippage = ((self.price_at_trade - present) / present) * 100

        return theoretical_rewards

    def refresh_basic():
        # HDF5_STORE basic.append(self.basic[-1])
        basic.drop(basic.index[0], inplace=True)

            

# i'll run a version of this on a cloud computer, too. the only difference is, it will wait a bit
# before saving. I think I'd prefer that it saves to storage synced with where my cpu saves
def start_time_loop(agent, td_env):

    from datetime import datetime, timedelta
    from threading import Timer

    # start next minute, at the 25 second mark
    now = datetime.today()
    start_time = now.replace(day=now.day, hour=now.hour, minute=now.minute, second=25, microsecond=0) + timedelta(minutes=1)
    
    delta_t = y - x
    secs = delta_t.total_seconds()

    t = Timer(secs, minute_loop, args=[agent, td_env])
    t.start()

def time_loop(agent, td_env):
    
    start_time = time.time()
    time_info = datetime.now()
    
    while True:
        # midwest trading h ours are 8:30am to 3pm
        # I might take out the if and have a bash script track that
        if time_info.hour + time_info.minute/100 >= 8.3 and \
        time_info.hour + time_info.minute/100 <=15:
            get_event_make_trade(agent, td_env)
            time.sleep(60.0 - ((time.time() - start_time) % 60.0))
            time_info = datetime.now()
    
# this is the live version
def get_event_make_trade(agent, td_env):
    
    event = {}
    
    event['experience'] = td_env.get_and_process_td_quote()
    event['action'] = agent.choose_action(event['experience'])
    td_env.execute_trade(event['action'])
    
    # reward can't be calculated until the next experience is gained, so reward is always one
    # step behind experience and action. and we get array of rewards for training
    # right now I'm just running with theoretical reward, not accounting for slippage
    event['reward'] = td_env.calc_reward()
    event['timestamp'] = self.basic['timestamp'][-1]
    agent.write_event(event)
    td_env.refresh_basic()

def get_n_mask_random(full_size, n):
        a_set = np.zeros((full_size))
        a_set[:n] = 1
        np.random.shuffle(a_set)
        return a_set 
    
def get_fresh_submask(full_mask, counter, len_sub):
    start = counter % len(full_mask)
    if start + len_sub > len(full_mask):
        
        mask = full_mask[start:]
        np.random.shuffle(full_mask)

        len_overflow = (start + len_sub) - len(full_mask)
        overflow = train_yes_set[0:len_overflow]
        mask = np.hstack(mask, overflow)
    
    else:
        end = start + len_sub
        mask = full_mask[start:start]
    
    counter += len_sub
    return sub_mask, full_mask, counter

def initialize_train_info(beliefs):
    print('no train_indices found, starting fresh')
    train = {
        'indc': {'train': [], 'val': []},
        'ratio': tv['train']['size'] / tv['val']['size'],
        'mask': get_n_mask_random(train_size + val_size, train_size),
        'count': 0,
    }
    event_store = {
        'experience': {},
        'action': [],
        'reward': [],
        'timestamp': []
    }
    if 'zarr_pics' in data_of:
        pics_shape = (len(indices[s]), 3, 224, 224)
        event_store['experience']['zarr_pics'] = make_array(pics_shape, 
                                                                f'event_store/{folder}/zarr_pics',
                                                                **zarr_kwargs)
    
    pickle.save(f'{beliefs.folder}/beliefs_dict.pk')

def online_agent_train(cfg):
    
    agent = Agent(cfg.run_space)

    # this and stats and event_store will all be saved in the model dict.
    # But they won't be put on the model, cause typing agent.beliefs... is too much
    try: 
        history = pickle.load(f'{agent.beliefs.folder}/beliefs_dict.pk')
        train = history['train']; event_store = history['event_store']
    except FileNotFoundError:
        train, event_store = start_beliefs_history(beliefs, folder)

    # reward will be shorter than the others because it is delayed by one timestep
    new_indc = np.array([i for i in range(agent.model.len_last_train, len(event_store['reward']))])

    sub_mask, train['mask'], train['count'] = get_fresh_submask(train['mask'], train['count'],
                                                            len(new_indc))
    
    new_train = list(new_indc[sub_mask == 1]); new_val = list(new_indc[sub_mask == 0])
    train['indc']['train'].extend(new_train); train['indc']['val'].extend(new_val)    
    
    sets = {}
    for s in train[indc]:
        train[indc][s] = train[indc][s][-tv[s]['size']]
        
        for p in event_store:
            if p == 'experience':
                sets[p] = {}
                for e in event_store[p]:
                    sets[p][e] = event_store[p][e][indices[i]]
            
            elif p == 'reward': # reward, maybe timestamp
                sets[p] = event_store[p][indices[i]]


    datasets = set_stuff.get_Xy(sets)
    
    env = Objdict()
    for a_set in datasets: env[a_set] = Env_Simple_Multi(live_data[a_set].X, live_data[a_set].y)
    
    agent, stats = act_and_learn(agent, env, num_episodes=1)


def append_stats(all_stats, stats):
            for a in all_stats:
                for b in all_stats[a]:
                    try: all_stats[a][b] = np.vstack(all_stats[a][b], stats[a][b])
                    except AttributeError:
                        for c in all_stats[a][b]:
                            all_stats[a][b][c] = np.vstack(all_stats[a][b][c], stats[a][b][c])

def initialize_all_stats(stats):
    all_stats = {}
    for a in stats:
        all_stats[a] = {}
        for b in stats[a][b]:
            
            if isinstance(stats[a][b], dict):
                all_stats[a][b] = {}
                for stat in stats[a][b][c]:
                    all_stats[a][b][c] = stats[a][b][c]
            else: all_stats[a][b] = stats[a][b]


def simulate_online(sets, tv, learn, zarr_kwargs, phases=('tv', 'test')):    

    for t_set in tv:
        try:
            tv[t_set].size = utils.get_sample_size(me=tv[t_set].size.me,
                                                    ci=tv[t_set].size.ci)
        except AttributeError: pass

    tv.train.last_len = 0
    memory_size = tv.train.size + tv.val.size
    
    t_sets = {
        'tv': tv,
        'test': {'test': {}}
    }

    # do I want to save everything for the simulation like I'm going to for the live
    # version? I think so...
    
    # initialize train and val indices
    train_yes_set = get_n_mask_random(memory_size, train_size)
    all_indices = [i for i in range(memory_size)]

    tv['train']['indc'] = list(all_indices[train_yes_set == 1])
    tv['val']['indc'] = list(all_indices[train_yes_set == 0])
    np.random.shuffle(train_yes_set)

    train_yes_counter = 0

    first_set = list(sets)[0]; first_cat = list(sets[first_set])[0]
    len_sets = len(sets[first_set][first_cat])
    steps = [i for i in range(memory_size, len_sets, train_every)]

    all_stats = {}
    
    for i in steps:
        for phase in phases:
            
            if phase == 'tv':
                try:
                    
                    # dictify the parameters and output and all that stuff
                    # look to real online function for inspiration
                    def extend_tv_indices(tv, train_yes_set, train_yes_counter):
                        new_indices = range(tv['last_len'], len(all_stats['test']['reward']))

                        train_yes, train_yes_set, train_yes_counter =\
                            get_fresh_submask(train_yes_set, train_yes_counter, len(new_indices))
                            
                        new_train = list(new_indices[train_yes == 1])
                        new_val = list(new_indices[train_yes == 0])
                        tv['train']['indc'].extend(new_train); tv['val']['indc'].extend(new_val)

                        return tv, train_yes_set, train_yes_counter

                except KeyError: pass

            # most of everything is just loading up the data for running
            for t_set in t_sets[phase]:
                
                # get the batches
                if phase == 'test': batch = (i, len_sets, train_every)
                elif phase == 'tv': batch = tv[t_set]['indc'][-tv[t_set]['size']:]

                # get the 
                for a_set in sets:
                    for cat in sets[a_set]:
                        if cat[0:4] != 'zarr':
                            t_sets[phase][t_set][cat] = sets[a_set][cat][batch]

                        else: # just zarr things.

                            t_path = f'simulation_zarr/{t_set}'
                            for i in range(2):
                                try:
                                    t_sets[phase][t_set][cat] = zarr.open(f"{t_path}/{cat}", mode='r')
                                    t_sets[phase][t_set][cat][:] = sets[a_set][cat][batch]; break
                                except (ValueError, IndexError) as error:
                                    Path(t_path).mkdir(parents=True, exist_ok=True)

                                    t_shape = sets[a_set][cat][batch].shape
                                    sets[phase][t_set][cat] = \
                                        make_array(t_shape, f"{t_path}{cat}", **zarr_kwargs)
                            
                            else: print("zarr_array didn't load right")

                
            varbs = set_stuff.get_Xy(t_sets['phase'])
            env = Objdict()
            for t_set in varbs: env[t_set] = Env_Simple_Multi(varbs[t_set].X, varbs[t_set].y)

            num_episodes = train_episodes if phase == 'tv' else 1
            agent, stats = act_and_learn(agent, env, num_episodes=num_episodes)
            agent.timestamps_trained = sets[first]['timestamp'][tv['train']['indc'][-train_size:]]
            
            try: all_stats[phase] = append_stats(all_stats[phase], stats)
            except KeyError: all_stats[phase] = initialize_all_stats(stats)

    return agent, all_stats



                
# I'll still need zarr even with more memory, because those pictures would take up a RIDICULOUS
# amount of memory if they weren't compressed by zarr. I might get away without zarr only if I'm
# able to figure something out for the pictures to where they're just 3 * 208
# I just need to make a model that's good enough.


    event['experience'] = td_env.get_and_process_td_quote()
    event['action'] = agent.choose_action(event['experience'])
    td_env.execute_trade(event['action'])
    
    # reward can't be calculated until the next experience is gained, so reward is always one
    # step behind experience and action. and we get array of rewards for training
    # right now I'm just running with theoretical reward, not accounting for slippage
    event['reward'] = td_env.calc_reward()
    event['timestamp'] = self.basic['timestamp'][-1]

    act_and_learn(agent, test_env, num_episodes=1, test_every=1)

    
    event_store = pickle.load('event_store.pk')
    indices = {}

    try: indices['train'] = pickle.load("train_indices.pk")
    except FileNotFoundError:
        indices['train'] = []; print('no train_indices found, starting fresh')
    
    # reward will be shorter than the others because it is delayed by one timestep
    len_current = len(event_store['reward'])
    len_last = agent.model.len_last_train
    new_indices = [i for i in range(len_last, len_current)]
    new_indices.shuffle
    
    # sort half into train and half into val
    # this method is not truly random, but pretty random
    split = len(new_indices) // 2
    if random.randint(0, 1) == 0: indices['train'].extend(new_indices[:split])
    else: indices['train'].extend(new_indices[-split:])
    indices['train'].sort()

    indices['val'] = np.setdiff1d(range(len(event_store['reward'])), indices['train'])

    sets = {}
    for i in indices:
        indices[i] = indices[i][-sample_size]
        
        for p in event_store:
            if p == 'experience':
                sets[p] = {}
                for e in event_store[p]:
                    sets[p][e] = event_store[p][e][indices[i]]
            
            elif p == 'reward':
                sets[p] = event_store[p][indices[i]]

    live_data = get_Xy(sets)
    
    env = Objdict()
    for a_set in live_data: env[a_set] = Env_Simple_Multi(live_data[a_set].X, live_data[a_set].y)

    
    agent = Agent(cfg.run_space)

    # I could for loop this for a certain amount of time
    agent = act_and_learn(agent, env, num_episodes=1, test_every=1)


    # when it's time to train the model
    # in practice, a different script will be responsible for this.
    utc = dt.datetime.utcfromtimestamp(new_event.timestamp)
    if utc.weekday() == 4 and utc.hour == 15 and utc.minute == 59:
        index = len(event_store['reward'])    



if __name__ == '__main__':
    
    initialize_event_store(cfg.data_of)
    td_env = Online_TD_Env(cfg.space.mungeprocess)
    agent = Agent(cfg.space.run)
    start_time_loop(agent, td_env)

# live stuff

    # combine both configs into one
    # combine mains into one
    
    # TD_Env init: accesses basic with HDF5 Store; puts just enough in memory to make a single
    # datapoint with one more addition

    # execute_trade(): a straightforward function with three options and some logic with each one
    # get_quote(): does what it says. I really want to figure out how to get a candlestick
    
    # preprocess: takes a raw datapoint and turns it into a basic datapoint
    # get_experience: turns basic datapoint into a regular pic and a context datapoint