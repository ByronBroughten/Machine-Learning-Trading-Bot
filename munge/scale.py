import torch, pickle, joblib, sklearn, os, zarr, shutil
import numpy as np
import datetime as dt
from pathlib import Path
from torchvision import transforms

import utils, munge

def load_scaler_params(package, scaler_type, params):
    if package == 'sklearn':
        if scaler_type == 'RobustScaler':
            
            assert all(p in ('center_', 'scale_') for p in params)
            scaler = getattr(sklearn.preprocessing, scaler_type)()
            for p in params:
                setattr(scaler, p, params[p])

    return scaler

class Transform(object):
    def __init__(self, scalers_infos, category):
        print('initializing Transform func')

        for i, si in enumerate(scalers_infos):
            if isinstance(si['to_transform'], str): # The truth value of an array is ambiguous
                if si['to_transform'] == 'same_as_fit':
                    scalers_infos[i]['to_transform'] = si['to_fit']
            
            print("to_fit: ", si['to_fit'])
            print("to_transform: ", si['to_transform'])

        self.scalers_infos = scalers_infos
        self.category = category

        scale_funcs = {
            'seqs': self.scale_seqs,
            'context': self.scale_context
        }
        self.scale_func = scale_funcs[category]
    
    def __call__(self, data):

        # setup
        squeeze = False
        if self.category == 'seqs':
            
            full_shape = data.shape
            data = munge.process.add_axis_if_needed(data, self.category)
            if data.shape < full_shape:
                squeeze = True

            continuity = 0
        
        # To make this more generalizable and legible, I could make continuity be a kwargs dict with labels.
        elif self.category == 'context':
            continuity = []

        
        # scaling
        for si in self.scalers_infos:
            data, continuity = self.scale_func(data, continuity, si)
            
        
        # wrapup
        if self.category == 'context':
            data = np.concatenate(continuity, 1)
        
        if squeeze:
            data = np.squeeze(data, axis=0)
        
        return data
    
    def scale_seqs(self, data, scaled, si):
        up_to = scaled + len(si['to_transform'])

        # for the scaler to fit, we gotta reshape np_sub
        data_sub = data[:, scaled:up_to, :]
        
        # reshape to fit scaler
        shape = data_sub.shape
        # print('shape', shape)
        data_sub = data_sub.reshape(shape[0], shape[1] * shape[2])
        data_sub = si['scaler'].transform(data_sub)

        # Put it back together.
        data[:, scaled:up_to, :] = data_sub.reshape(*shape)

        scaled += len(si['to_transform'])

        return data, scaled
    
    def scale_context(self, data, transformed, si):
        print("len(data)", len(data))
        transformed += [si['scaler'].transform(data[si['to_transform']])]
        print("len(data)", len(data))
        return data, transformed


def get_zarr_quartiles(data, ratio, mode=None, indices=None):
    
    if mode == 'seqs':

        fold = 'temp_scaler_sets'
        Path(fold).mkdir(parents=True, exist_ok=True)
        
        # this is where if I were trying real hard I could read the memory capacity of the
        # computer and split up the data based on that. But I'll just settle on 200000

        arrs = []
        for i in range(0, len(data), 200000):
            
            try: # avoid making new array
                np_sub[:] = data.oindex[i:i+20000, indices, :]
                break
            except (ValueError, NameError): # make new array
                np_sub = data.oindex[i:i+20000, indices, :]
            
            np_shape = np_sub.shape
            np_sub = np_sub.flatten('f')
            
            path = os.path.join(fold, 'zarr' + str(i))
            zarr_sub = utils.zarrs.make_array(np_sub.shape, path, chunk=np_sub.shape[0], clevel=3)
            zarr_sub[:] = np_sub
            arrs.append(zarr_sub)

            np_sub = np_sub.reshape(np_shape)
        
        quartile = utils.stats.get_quartile_from_arrs(arrs, ratio)
        shutil.rmtree(fold, ignore_errors=True)
    
    return quartile

def get_sk_scaler(data, category, scaler_type, to_fit):
    
    # I can add different sources of scalers, such as those for pytorch pics.
    scaler = getattr(sklearn.preprocessing, scaler_type)()

    if category == 'seqs':
        if scaler_type == 'RobustScaler':
            
            third_quartile = get_zarr_quartiles(data, .75, mode='seqs', indices=to_fit)
            first_quartile = get_zarr_quartiles(data, .25, mode='seqs', indices=to_fit)
            scaler.scale_ = third_quartile - first_quartile
            scaler.center_ = get_zarr_quartiles(data, .5, mode='seqs', indices=to_fit)

    elif category == 'context':
    
        # turn dataframe into numpy array to make sure dimensions are right
        data = data[to_fit].values
        if len(data.shape) < 2:
            data = data[:, np.newaxis]

        scaler.fit(data)
    return scaler

def make_scalers(scalers, data, glob_kind):
    print("I'm prepping for scaling.")

    for i, sc in enumerate(scalers):

        if 'scaler' not in sc:
            print("I'm making a scaler.")
            scaler = get_sk_scaler(data, glob_kind, sc['obj_name'], sc['to_fit'])
            
            if sc['obj_name'] == 'RobustScaler':
                scalers[i]['params'] = {
                    'center_': scaler.center_,
                    'scale_': scaler.scale_
                }
        
            Path(os.path.dirname(sc['path'])).mkdir(parents=True, exist_ok=True)
            print("I'm saving the scaler to this path:\n", sc['path'])
            joblib.dump(scaler, sc['path'])
            scalers[i]['scaler'] = scaler
            

    return scalers

def finish_scaler_path(path, obj_name, to_fit, glob_kind, folder=None, seq_len=None):

    dir_name = os.path.dirname(path)
    base_name = os.path.basename(path)

    scaler_name = obj_name
    scaler_name += f"-{base_name}"

    if glob_kind == 'seqs':
        scaler_name += f"-{seq_len}"
    
    for feature in to_fit:
        scaler_name += f"-{feature}"
    
    path = os.path.join(dir_name,  scaler_name)
    return path

def get_what_fit(scalers, data, glob_kind):
    for i, sc in enumerate(scalers):
        if glob_kind == 'seqs':
            
            for _ in range(2):
                try:
                    sc['to_fit'] = [i for i in range(cols_fitted, cols_fitted + len(sc['to_fit']))]
                    break
                except UnboundLocalError:
                    cols_fitted = 0
            else: raise UnboundLocalError("There might be an unbound local here besides cols_fitted.")
    
            cols_fitted += len(sc['to_fit'])
            cols_fitted = len(sc['to_fit'])
        
        if glob_kind == 'context' and isinstance(sc['to_fit'], str): # the truth value of an array is ambiguous
            if sc['to_fit'] == 'all_as_is':
                scalers[i]['to_fit'] = data.columns
    
    return scalers

def load_scalers(scalers, path, glob_kind, is_train, folder=None):
    print('Attempting to load one or more scalers.')
    
    for i, sc in enumerate(scalers):

        if 'path' in sc:
            scaler_path = sc['path']

        else: # If there's no path we'll have to make one.
            
            path_kwargs = {
                'path': path,
                'glob_kind': glob_kind,
                'obj_name': sc['obj_name'],
                'to_fit': sc['to_fit'],
            }
            # if folder: path_kwargs['folder'] = folder
            if 'seq_len' in sc:
                path_kwargs['seq_len'] = sc['seq_len']

            scalers[i]['path'] = finish_scaler_path(**path_kwargs)
            scaler_path = scalers[i]['path']

            try:
                scalers[i]['scaler'] = joblib.load(scaler_path)
            except FileNotFoundError:
                print(f"I couldn't load a scaler from {scaler_path}.")
                if not is_train:
                    raise FileNotFoundError("And that's especially a shame because this is isn't a training set.")
                # elif folder != os.path.dirname()
        
    return scalers

# class Scaler_Maker_Saver():
#     def __init__(self, info, label):
#         self.train_len = info.train_len
#         self.index = 0
#         self.what_fit = info.what_fit
#         self.label = label + '-' + info.what_fit
    
#     def gather_data(self, sets):
#         print('gathering_data_for_scaler')
#         for s in sets:
#             if s[0:5] == 'train':
#                 for d, v in sets[s].items():
#                     if d == self.what_fit or d in self.what_fit:
#                         print('s, v.shape')
#                         print(s, v.shape)
                        
#                         try: self.set_to_fit[self.index : self.index+len(v)] = v
#                         except AttributeError:
#                             dims = (self.train_len, *v.shape[1:])
#                             self.set_to_fit = np.zeros(dims)
#                             self.set_to_fit[self.index : self.index+len(v)] = v
                        
#                         self.index += len(v)
        
#         if self.index == len(self.set_to_fit):
#             print('all data gathered for scaler')
        
#     def make_and_save_robust_scaler(self):
#         scaler = RobustScaler()
#         scaler.fit(self.set_to_fit)
#         joblib.dump(scaler, f'scalers/{self.label}')
#         print('scaler saved')