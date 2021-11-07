import time, math, json, os, shutil, operator, torch
import numpy as np
import datetime as dt


class Objdict(dict):
    def __init__(self, dic=None):
        super(Objdict, self).__init__()
        if dic != None:
            try:
                for k, v in dic.items():
                    if isinstance(v, dict):
                        self[k] = Objdict(v)
                    else: self[k] = v
            except TypeError:
                raise TypeError('Objdict only accepts objects with key-value pairs, namely dicts')
    
    def __setitem__(self, key, value):
        try: value = Objdict(value)
        except (AttributeError, ValueError): pass
        super(Objdict, self).__setitem__(key, value)

    def __setattr__(self, name, value):
        try: value = Objdict(value)
        except (AttributeError, ValueError): pass
        self[name] = value
    
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)

class Get_Webdriver(object):
    def __init__(self, webdriver_path):
        self.webdriver_path = webdriver_path
    
    def __call__(self):
        from selenium import webdriver
        driver = webdriver.Chrome(executable_path=self.webdriver_path)
        return driver

class Pack_Runner(object):
# this will work as expected so long as the globals in the module where this is initialized are all 
# declared before this is called. If I didn't want this to depend on the globals where it's initialized,
# I could have it dynamically import packages, but then my imports wouldn't be so explicit.
    def __init__(self, globals_dict):
        self.globals = globals_dict

    def __call__(self, pack, varbs_dict, *args, dry_run=False, **kwargs):

        args = list(args)

        if 'args' in pack: args = list(args) + pack['args']
        if 'kwargs' in pack: kwargs.update(pack['kwargs'])

        if 'varbs' in pack:
            args_from_dict, kwargs_from_dict = get_args_and_kwargs(pack['varbs'], varbs_dict)

            args += args_from_dict
            kwargs.update(kwargs_from_dict)

        package_func = pack['func'].split('.', 1)
        package_name = package_func[0]

        try:
            func_name = package_func[1]
        except IndexError:
            raise IndexError(f"pack['func'] {pack['func']} doesn't seem to be a module with an attribute.")

        try:
            package = self.globals[package_name]
        except KeyError:
            print('\nPrint')
            print('global keys:', self.globals.keys())
            print('attempted package:', package_name)
            print()
            raise KeyError('See print.')
        
        
        func = operator.attrgetter(func_name)(package)
        # except AttributeError:
        #     print(f"Apparently {func_name} is not in {package_name}.")
        #     raise AttributeError('See print.')

        if dry_run:
            result = None

        elif not dry_run:
            print('\nfunc:', pack['func']); start = dt.datetime.now()
            result = func(*args, **kwargs)
            print(f'func time taken: {dt.datetime.now() - start}')

        return result

# def change_torch_data_type(torch_thing, data_type):
#     if mode == 'torch':
#         if data_type == 'float':
#             thing = thing.float()
#         elif data_type == 'double':
#             thing = thing.double()
#         elif data_type == 'half':
#             thing = thing.half()

#     return thing
# def change_data_type(thing, data_type, mode='np'):

#     if mode == 'np':
#         if data_type == 'float':
#             thing = thing.astype(np.float32)
#         elif data_type == 'double':
#             thing = thing.astype(np.float64)
#         elif data_type == 'half':
#             thing = thing.astype(np.float16)

def get_dtype_np_torch(dtype_str, package='np'):
    dtype = getattr(globals()[package], dtype_str)
    return dtype

def get_args_and_kwargs(varb_lists, varbs_dict):

    if not 'args' in varb_lists and not 'kwargs' in varb_lists:
        print(varb_lists); raise KeyError('Neither args nor kwargs are in varb_lists. See print.')

    args = []
    kwargs = {}    
    
    try:
        for args_or_kwargs, groups, in varb_lists.items():
            
            if args_or_kwargs == 'args':
                for group in groups:
                    for group_name, arg_list in group.items():
                        for varb_name in arg_list:
                            args.append(varbs_dict[group_name][varb_name])

            elif args_or_kwargs == 'kwargs':
                for group_name, kwargs_list in groups.items():
                    for varb_name in kwargs_list:
                        kwargs[varb_name] = varbs_dict[group_name][varb_name]
    except KeyError:
        raise KeyError(f"varbs_dict group {group_name} doesn't have {varb_name}. Here, look: {varbs_dict}")
    
    return args, kwargs
    # except KeyError:
    #     print('I failed with the following.')
    #     print('key_word:', key_word, ' group_name:', group_name, '\nkwargs_group keys:', kwargs_group.keys())
    #     raise KeyError('See print.')

def json_to_dict(path):
    with open(path, 'r') as json_file:
        dictionary = json.load(json_file)
    return dictionary

# math
def trunc(num, dec=0):
    stepper = 10.0 ** dec
    return math.trunc(stepper * num) / stepper

# time
def chop_microsecs(delta):
    return delta - dt.timedelta(microseconds=delta.microseconds)

def stamp_mils_to_secs(num_or_arr):
    num_or_arr = num_or_arr // 1000
    return num_or_arr

def dt_to_time(t):
    return time.mktime(t.timetuple()) + t.microsecond / 1E6

#arrays
def cat_len(list_of_matrices):
    return len(np.concatenate(list_of_matrices))

def shuffle(array):
    return np.random.permutation(array)


def func_in_chunks(arr, chunk_size, func, kwargs=None):
    if kwargs is None: kwargs = {}
    
    print(f"Funcin' in chunks with {str(func)}")
    for i in range(0, len(arr), chunk_size):
        start = dt.datetime.now()
        print(i, end=" ")
        arr[i:i+chunk_size] = func(arr[i:i+chunk_size], **kwargs)
        print(dt.datetime.now() - start)
    return arr

def get_chunky(c, len_arr, chunk_size):
# possibly pointless because trying to slice beyond len_arr is allowed
    if c + chunk_size > len_arr:
        chunk = None
        print('rest of set')
    else:
        chunk = c + chunk_size
    return chunk

# strings
def split_text_where(text, char, n=1):
    parts = text.split(char)
    split_text = char.join(parts[:n]), char.join(parts[n:])
    return split_text

def join_string_with_what(string_tuple, char=''):
    string = ''
    for i in range(len(string_tuple)):
        string = string + string_tuple[i]
        if i != len(string_tuple) - 1:
            string = string + char
    
    return string

def str_lst_replace_char(a_list, char, rep=''):
    a_list = [item.replace(char, rep) for item in a_list]
    return a_list

def string_from_keylists_dic(dic, lists):
    string = ''
    for ls in lists:
        result = get_idx_vals(dic, ls)
        if isinstance(result, list):
            for rs in result:
                string += str(rs) + '-'
        else:
            string += str(result) + '-'
    
    string = string[:-1]
    
    return string

def get_idx_vals(iterable, key_list):
    for key in key_list:
        if isinstance(key, list):
            iterable = string_from_keylists_dic(iterable, key)
        else: 
            try:
                iterable = iterable[key]
            except(IndexError, TypeError, KeyError):
                print('\niterable:', iterable)
                print('\nkey:', key)
                raise ValueError('See print.')
        
    return iterable

def get_dict_string(dic, items):
    string = get_dict_string_inner(dic, items)
    return string[:-1]

def get_dict_string_inner(dic, items):
# a dictionary and a tuple of strings and tuples of strings is passed in

    string = ''

    for i in items:
        
        # if the item is itself a tuple, then the dict is passed that tuple's first
        # item, to go to that layer of the dict, along with the tuple of items for that next layer
        if isinstance(i, list):
            string += get_dict_string_inner(dic[i[0]], i[1])
            # string = f'{string}{get_dict_string(dic[i[0]], i[1])}'
        else:
            # if the item in the tuple of strings isn't a tuple, it is used to index the dict
            if isinstance(dic[i], list):
                extension = ''
                for item in dic[i]:
                    extension += str(item) + '-'                
            else: 
                extension = str(dic[i]) + '-'

            string += extension
            # string = f'{string}{str(dic[i])}-'
            
            string = string.replace('()', '')
    
    return string