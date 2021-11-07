import torch, os
import torch.nn.functional as F
import torchvision.models as models
from torch import nn
import torch.optim as optim
import datetime as dt
from train.model_blocks import TCN, Dense_Net, Dense_Layer
import utils


def get_custom_model(make, model_kwargs, dense_info=None, pretrained=None):    
    
    # for models that use pretrained (or not) pytorch models
    if pretrained is not None: model_kwargs['pretrained'] = pretrained
    
    # get the custom model
    custom_models = Custom_Models()
    model = getattr(custom_models, make)(**model_kwargs)

    # for pytorch resnext--I could change the model names and make this based
    # on that, that would be better
    # if pretrained != None: model.fc = nn.Linear(2048, y_dim)
    return model
        
def make_model_name(make, space):
        characteristics = utils.misc.get_dict_string(space.__dict__, space.model_name_varbs)
        birth_time = dt.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        name = utils.misc.join_string_with_what((make, characteristics, birth_time), char='|')
        return name

def get(space, make, model_kwargs, dtype, device=None):
        
    print('making new model')
    model = get_custom_model(make, model_kwargs)
    model.optimizer = optim.Adam(model.parameters())
    model.name = make_model_name(make, space) 

    package = {
        'name': model.name,
        'make': make,
        'opt_make': 'Adam',
        'kwargs': model_kwargs,
        'score': 0,
        'history': []
    }

    print(f'your new baby model is named {model.name}\n')
    model = model.type(dtype)
    
    return model, package

    #Test
    # import torch.optim as optim
    # import torch.nn as nn
    # import torch.nn.functional as F

    # class Generic(nn.Module):
    #     def __init__(self):
    #         self.fc1 = nn.Linear(10, 3)
    #     def forward(self, obs):
    #         x = F.relu(self.fc1(obs))
    #         return x
    # generic = Generic()
    # optimizer = optim.Adam(generic.parameters())
    # print(optimizer)


class Custom_Models(object):

    class TCN_FC_1D(torch.nn.Module):
        def __init__(self, TCN_kwargs, context_kwargs, final_kwargs, output):
            super(Custom_Models.TCN_FC_1D, self).__init__()

            # sequence model
            self.TCN = TCN(**TCN_kwargs)
            self.context = Dense_Net(**context_kwargs)
            self.final = Dense_Net(**final_kwargs)
            
            self.output = nn.Linear(final_kwargs['out_size_arr'][-1], output)

            self.apply(self.init_all_weights)
            

        def init_all_weights(self, m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight)

        def forward(self, X):

            for x_kind, x in X.items():

                if x_kind == 'seqs':
                    x0 = self.TCN(x)

                elif x_kind == 'context':
                    x1 = self.context(x)

            x2 = torch.cat((x0, x1), 1)
            x2 = self.final(x2)
            return self.output(x2)


    #%%
    class DDQ_FC_Resnext50(torch.nn.Module):
        def __init__(self, dense_layer_dict, last_dense_size, y_dim):
            super(DDQ_FC_Resnext50, self).__init__()

            # get resnext 50
            self.model = models.resnext50_32x4d(pretrained=True)

            # make extra fc layer to tack onto the end of it
            self.pic0_fc = nn.Linear(2048, 2048)

            self.pic0_bn = nn.BatchNorm1d(2048)
            self.pic0_in = nn.InstanceNorm1d(2048)

            self.pic0_act = nn.ReLU()
            self.dropout = nn.Dropout(0.3)

            # make fc layers for the context model
            for key in dense_layer_dict:
                setattr(self, key, dense_layer_dict[key])
            self.dense_layer_attrs = list(dense_layer_dict.keys())
            
            # output based on combined models, replaced output layer of non-q model
            self.V = nn.Linear(2048 + last_dense_size, 1)
            self.A = nn.Linear(2048 + last_dense_size, y_dim)
        
        def forward(self, X):
        # X is a list of two items: pics and context variables

            # PRETRAINED VISION MODEL
            # resnext50
            # feed it picture data, aka X[0]
            for i in X:

                if i[-4:] == 'pics':
                    
                    # print('X[i]', i, '\n', X[i])
                    x0 = self.model.conv1(X[i])

                    # pass x0 through the pretrained resnext layers
                    x0 = self.model.bn1(x0)
                    x0 = self.model.relu(x0)
                    x0 = self.model.maxpool(x0)

                    x0 = self.model.layer1(x0)
                    x0 = self.model.layer2(x0)
                    x0 = self.model.layer3(x0)
                    x0 = self.model.layer4(x0)
                    x0 = self.model.avgpool(x0)

                    # one dense layer pass-through to liken it to the dense model
                    x0 = x0.view(x0.size(0), -1)
                    x0 = self.pic0_fc(x0)
                    x0 = self.pic0_bn(x0)
                    x0 = self.pic0_act(x0)
                    x0 = self.dropout(x0)

                elif i[-7:] == 'context':
                    
                    # print('X[i]', i, '\n', X[i])

                    # DENSE CONTEXT MODEL
                    # fresh fully-connected model, dynamically generated and executed
                    # get the first dense layer, and feed it context data, aka X[1]
                    x1 = getattr(self, self.dense_layer_attrs[0])(X[i])
                    
                    # pass x1 through the rest of the dense layers
                    for attr in self.dense_layer_attrs[1:]:
                        x1 = getattr(self, attr)(x1)

            # COMBINE
            x2 = torch.cat((x0, x1), 1)
            
            # output based on combined models, replaced output layer for V and A
            # value = self.V(x2)
            advantage = self.A(x2)
            return advantage

    class dual_resnext50_pt_fc(torch.nn.Module):
        def __init__(self, dense_layer_dict, last_dense_size, y_dim):
            super(dual_resnext50_pt_fc, self).__init__()

            # get resnext 50
            self.model = models.resnext50_32x4d(pretrained=True)

            # make extra fc layer to tack onto the end of it
            self.pic0_fc = nn.Linear(2048, 2048)
            self.pic0_bn = nn.BatchNorm1d(2048)
            self.pic0_act = nn.ReLU()
            self.dropout = nn.Dropout(0.3)

            # make fc layers for the context model
            for key in dense_layer_dict:
                setattr(self, key, dense_layer_dict[key])
            self.dense_layer_attrs = list(dense_layer_dict.keys())
            
            # output based on combined models
            self.output = nn.Linear(2048 + last_dense_size, y_dim)
        
        def forward(self, X):
        # X is a list of two items: pics and context variables

            # PRETRAINED VISION MODEL
            # resnext50
            # feed it picture data, aka X[0]
            for i in X:

                if i[-4:] == 'pics':
                    x0 = self.model.conv1(X[i])

                    # pass x0 through the pretrained resnext layers
                    x0 = self.model.bn1(x0)
                    x0 = self.model.relu(x0)
                    x0 = self.model.maxpool(x0)

                    x0 = self.model.layer1(x0)
                    x0 = self.model.layer2(x0)
                    x0 = self.model.layer3(x0)
                    x0 = self.model.layer4(x0)
                    x0 = self.model.avgpool(x0)

                    # one dense layer pass-through to liken it to the dense model
                    x0 = x0.view(x0.size(0), -1)
                    x0 = self.pic0_fc(x0)
                    x0 = self.pic0_bn(x0)
                    x0 = self.pic0_act(x0)
                    x0 = self.dropout(x0)
                else: raise KeyError('no key "...pics" in X')
                

                if i[-7:] == 'context':
                    # DENSE CONTEXT MODEL
                    # fresh fully-connected model, dynamically generated and executed
                    # get the first dense layer, and feed it context data, aka X[1]
                    x1 = getattr(self, self.dense_layer_attrs[0])(X[1])
                    
                    # pass x1 through the rest of the dense layers
                    for attr in self.dense_layer_attrs[1:]:
                        x1 = getattr(self, attr)(x1)
                else: raise KeyError('no key "...context" in X')


            # combine and output
            x2 = torch.cat((x0, x1), 1)
            return self.output(x2)


    class DuelingLinearDeepQNetwork(nn.Module):
        def __init__(self, ALPHA, n_actions, name, input_dims, chkpt_dir='tmp/dqn'):
            super(DuelingLinearDeepQNetwork, self).__init__()

            self.fc1 = nn.Linear(*input_dims, 128)
            self.fc2 = nn.Linear(128, 128)
            self.V = nn.Linear(128, 1)
            self.A = nn.Linear(128, n_actions)

            self.optimizer = optim.Adam(self.parameters(), lr=ALPHA)
            self.loss = nn.MSELoss()
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            self.to(self.device)
            self.checkpoint_dir = chkpt_dir
            self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_dqn')

        def forward(self, state):
            l1 = F.relu(self.fc1(state))
            l2 = F.relu(self.fc2(l1))
            V = self.V(l2)
            A = self.A(l2)

            return V, A


    # for now don't use this--it'll be easier just to adjust everything else.
    class Binary_Model(torch.nn.Module):
        
        def __init__(self, last_dense_size, dense_layer_dict):
            '''The constructor (super) is where to set up the node and activation layers'''
            super().__init__()

            for key in dense_layer_dict:
                setattr(self, key, dense_layer_dict[key])
            self.dense_layer_attrs = list(dense_layer_dict.keys())
            
            self.output = nn.Linear(last_dense_size, 1)


        def forward(self, X):
            # begin
            thought = getattr(self, self.dense_layer_attrs[0])(X)
            
            for key in self.dense_layer_attrs[1:]:
                thought = getattr(self, key)(thought)

            # end
            thought = self.output(thought)
            return torch.sigmoid(thought, dim=-1)


    # for now only use this
    class Multi_Class_Model(torch.nn.Module):
        
        def __init__(self, y_dim, last_dense_size, dense_layer_dict):
            '''The constructor (super) is where to set up the node and activation layers'''
            super().__init__()

            for key in dense_layer_dict:
                setattr(self, key, dense_layer_dict[key])
            self.dense_layer_attrs = list(dense_layer_dict.keys())
            
            self.output = nn.Linear(last_dense_size, y_dim)


        def forward(self, X):
            
            # begin
            thought = getattr(self, self.dense_layer_attrs[0])(X)
            
            for key in self.dense_layer_attrs[1:]:
                thought = getattr(self, key)(thought)

            # end
            thought = self.output(thought)
            return torch.softmax(thought, dim=-1)


def get_simple_hidden_layer(n, outputs, hidden_layer_size=None):
    
    # I could make this more dynamic if I wanted to deal with more than two labels.
    # This could define different output sizes and perhaps hidden layer sizes for more hidden layers
    # This could also define different inputs for chains of nns

    # one hidden layer, nodes = mean (num features, num outputs)
    if hidden_layer_size == None:
        hidden_layer_size = int((n + outputs) / 2)
    
    return hidden_layer_size

