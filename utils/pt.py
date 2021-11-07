import torch
import numpy as np
import torch.cuda as cuda

from torch.utils.data import DataLoader

def list_of_tensors_to_numpy(list_of_tensors, count_to=None):
# this gets however much of the end of a list of tensors as you want into a numpy array

    array = []
    count = 0
    for i in range(len(list_of_tensors)):
        i = i + 1
        t = list_of_tensors[-i]
        t = t.to('cpu').numpy()
        t = t[::-1]

        array.append(t)

        if count_to != None:
            count += len(array)
            if count >= count_to: break
    
    array = np.concatenate(array)
    array = array[:count_to]
    return array

def get_device():
    
    if cuda.is_available():
        device = torch.device('cuda:0')
        device_name = cuda.get_device_name(0)
        print('running on gpu')

    else:
        device = torch.device('cpu')
        device_name = 'cpu'
        print('running on cpu')

        # print(torch.cuda.device_count())
        # torch.device('cuda:0' if torch.cuda.is_available() else 'cuda:1')
        # self.to(self.device)

        # maybe eventually try to get TPU to work
        # device = xm.xla_device()
        # device_name = 'tpu'

    return device, device_name

def check_device(name=None):
    if name is not None:
        device = torch.device(name)
        device_name = cuda.get_device_name(0)
    else:
        device, device_name = get_device()
    
    return device #, device_name

def get_loaders_train_and_val(trainset, valset, batch_size, val_batch_size):
    
    if len(trainset) >= batch_size:
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    else:
        print('len(trainset)', len(trainset))
        train_loader = DataLoader(trainset, batch_size=len(trainset), shuffle=True)


    if len(valset) >= batch_size:
        val_loader = DataLoader(valset, batch_size=val_batch_size, shuffle=False)
    else:
        print('len(valset)', len(valset))
        val_loader = DataLoader(valset, batch_size=len(valset), shuffle=False)
    

    dataloaders = {
        'train': train_loader,
        'val': val_loader    
    }

    return dataloaders

# translate numbers to predictions
def pred_sigmoid(output):
    y_pred = (output >= .5).type(torch.uint8) 
    return y_pred

def pred_tanh(output):
    y_pred = (output >= 0).type(torch.uint8) 
    return y_pred

def pred_multiclass(output):
    y_pred = torch.max(output, 1)[1]
    return y_pred

def pred_reg(output):
    return output

# functions for making predictions, depending on the type of activation function used on the output
def get_pred_func(pred_string):
    
    if pred_string == 'sigmoid': pred_func = pred_sigmoid

    elif pred_string == 'tanh': pred_func = pred_tanh

    elif pred_string == 'multiclass': pred_func = pred_multiclass
    
    elif pred_string == 'reg': pred_func = pred_reg
    
    return pred_func
