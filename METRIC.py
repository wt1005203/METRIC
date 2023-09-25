#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import math
import random
import matplotlib
import matplotlib.pyplot as plt
import os
import sys
import warnings
import copy
warnings.filterwarnings("ignore")
def figure_size_setting(WIDTH):
    #WIDTH = 700.0  # the number latex spits out
    FACTOR = 0.8  # the fraction of the width you'd like the figure to occupy
    fig_width_pt  = WIDTH * FACTOR
    inches_per_pt = 1.0 / 72.27
    golden_ratio  = (np.sqrt(5) - 1.0) / 2.0  # because it looks good
    fig_width_in  = fig_width_pt * inches_per_pt  # figure width in inches
    fig_height_in = fig_width_in * golden_ratio   # figure height in inches
    fig_dims    = [fig_width_in, fig_height_in] # fig dims as a list
    return fig_dims

cross_validation_number = sys.argv[1]
path_prefix_algorithm = sys.argv[2]
path_prefix_data = "/".join(path_prefix_algorithm.split("/")[:-2]) + "/processed_data/"
if os.path.exists(path_prefix_algorithm)==False:
    os.mkdir(path_prefix_algorithm)

# Load Data
X_train_ori = pd.read_csv(path_prefix_data + "X_train.csv", header=None).values;
y_train_ori = pd.read_csv(path_prefix_data + "y_train.csv", header=None).values;
X_test_ori = pd.read_csv(path_prefix_data + "X_test.csv", header=None).values;
y_test_ori = pd.read_csv(path_prefix_data + "y_test.csv", header=None).values;
print(X_train_ori.shape, y_train_ori.shape, X_test_ori.shape, y_test_ori.shape)

# establish X_train_all, y_train_all
import numpy.matlib
numOfRepeats = 1
std_list = [0.5, 1.0, 1.5]
for i_std, std in enumerate(std_list):
    X_train = X_train_ori.copy()
    y_train = X_train_ori[:,-y_train_ori.shape[1]:].copy()
    X_train = numpy.matlib.repmat(X_train, numOfRepeats, 1)
    y_train = numpy.matlib.repmat(y_train, numOfRepeats, 1)
    noises = np.random.normal(loc=0.0, scale=np.log(10**std), size=y_train.shape)
    X_train[:,-y_train.shape[1]:] = X_train[:,-y_train.shape[1]:].copy() + noises
    if i_std==0:
        X_train_all = X_train.copy()
        y_train_all = y_train.copy()
    else:
        X_train_all = np.concatenate([X_train_all, X_train], axis=0)
        y_train_all = np.concatenate([y_train_all, y_train], axis=0)
        
for i_std, std in enumerate(std_list):
    X_train = X_train_ori.copy()
    y_train = X_train_ori[:,-y_train_ori.shape[1]:].copy()
    X_train = numpy.matlib.repmat(X_train, numOfRepeats, 1)
    y_train = numpy.matlib.repmat(y_train, numOfRepeats, 1)
    noises = np.random.uniform(low=-np.sqrt(12)*np.log(10**std)/2, high=np.sqrt(12)*np.log(10**std)/2, size=y_train.shape)
    X_train[:,-y_train.shape[1]:] = X_train[:,-y_train.shape[1]:].copy() + noises
    X_train_all = np.concatenate([X_train_all, X_train], axis=0)
    y_train_all = np.concatenate([y_train_all, y_train], axis=0)

X_train = X_train_all.copy()
y_train = y_train_all.copy()

numOfRepeats = 1
for i_std, std in enumerate(std_list):
    X_test = X_test_ori.copy()
    y_test = X_test_ori[:,-y_test_ori.shape[1]:].copy()
    X_test = numpy.matlib.repmat(X_test, numOfRepeats, 1)
    y_test = numpy.matlib.repmat(y_test, numOfRepeats, 1)
    noises = np.random.normal(loc=0.0, scale=np.log(10**std), size=y_test.shape)
    X_test[:,-y_test.shape[1]:] = X_test[:,-y_test.shape[1]:].copy() + noises
    if i_std==0:
        X_test_all = X_test.copy()
        y_test_all = y_test.copy()
    else:
        X_test_all = np.concatenate([X_test_all, X_test], axis=0)
        y_test_all = np.concatenate([y_test_all, y_test], axis=0)
X_test = X_test_all.copy()
y_test = y_test_all.copy()
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

        
# ### Hyperparameter selection
#### import libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

def spearman(target, pred):
    pred = pred.detach().numpy()
    target = target.detach().numpy()
    df1 = pd.DataFrame(pred)
    df2 = pd.DataFrame(target)
    metabolites_corr = df1.corrwith(df2, axis = 0, method='spearman').values
    return np.nanmean(metabolites_corr)

from sklearn.preprocessing import StandardScaler 
scaler_x = StandardScaler()  
scaler_x.fit(X_train)  
X_train = scaler_x.transform(X_train)  
X_test = scaler_x.transform(X_test)  
scaler_y = StandardScaler()  
scaler_y.fit(y_train)  
y_train = scaler_y.transform(y_train)  
y_test = scaler_y.transform(y_test)  

#basic tools 
import os
import numpy as np
import pandas as pd
import warnings

#tuning hyperparameters
from bayes_opt import BayesianOptimization
from skopt  import BayesSearchCV 

#building models
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
import time
import sys
from sklearn.metrics import mean_squared_error
import copy

#metrics 
from sklearn.metrics import roc_auc_score, roc_curve
#import shap
warnings.filterwarnings("ignore")

N_input = X_train.shape[1]
N_output = y_train.shape[1]

def spearman(target, pred):
    pred = pred.detach().numpy()
    target = target.detach().numpy()
    df1 = pd.DataFrame(pred)
    df2 = pd.DataFrame(target)
    metabolites_corr = df1.corrwith(df2, axis = 0, method='spearman').values
    return np.nanmean(metabolites_corr)

class Feedforward(torch.nn.Module):
    def __init__(self, input_dim=N_input, hidden_dim=256, dropout=.5, alpha=1.0, beta=1.0, output_dim=N_output, Nlayers=1):
        super(Feedforward, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim  = hidden_dim
        self.output_dim = output_dim
        self.fc1 = torch.nn.Linear(self.input_dim, self.hidden_dim)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = torch.nn.Linear(self.hidden_dim, self.output_dim)
        # Define proportion or neurons to dropout
        self.dropout = torch.nn.Dropout(dropout)
        self.alpha = alpha
        self.beta = beta
        self.Nlayers = Nlayers
    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        relu_dropout = self.dropout(relu)
        for i in range(self.Nlayers-1):
            hidden = self.fc2(relu_dropout)
            relu = self.relu(hidden)
            relu_dropout = self.dropout(relu)
        output = self.fc3(relu_dropout)
        output = (1-self.alpha) * output + self.alpha * x[:,-self.output_dim:]
        return output
    
def training(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,layer_size=512,Nlayers=1,lr=0.001,l2=0.001,dropout=0.5,alpha=1.0,beta=1.0,batch_size=64,epochs=50):
    torch.manual_seed(0)
    losses=[]
    X_train, y_train, X_test, y_test= map(
    torch.FloatTensor, (X_train, y_train, X_test, y_test)
    )
    layer_size = int(round(layer_size))
    Nlayers = int(round(Nlayers))

    train_ds = TensorDataset(X_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=batch_size)
    model = Feedforward(hidden_dim=layer_size, Nlayers=Nlayers, dropout=dropout, alpha=alpha, beta=beta)
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=l2)
    test_error = []
    model_list = []
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            pred = model(xb)
            loss = loss_func(pred, yb)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        losses.append(loss)

        model.eval()
        y_pred = model(X_test)
        after_train = spearman(y_pred.squeeze(), y_test) 
        test_error = test_error + [after_train]
        model_list = model_list + [copy.deepcopy(model)]
    plt.plot(test_error)
    if np.where(test_error==np.max(test_error))[0].shape[0] > 0:
        best_model = copy.deepcopy(model_list[np.where(test_error==np.max(test_error))[0][0]])
    else:
        best_model = copy.deepcopy(model)
    best_model.eval()
    y_pred = best_model(X_test)
    mse = mean_squared_error(y_pred.detach().numpy(),y_test.detach().numpy())
    return -mse #to specify what to maximize in this function

def cross_validation(X_train=X_train_ori,y_train=y_train_ori,X_test=X_test_ori,y_test=y_test_ori,layer_size=512,Nlayers=1,lr=0.001,l2=0.001,dropout=0.5,alpha=1.0,beta=1.0,batch_size=64,epochs=50):
    from sklearn.model_selection import ShuffleSplit # or StratifiedShuffleSplit
    from sklearn.model_selection import KFold
    n_splits = 5
    kf = KFold(n_splits=n_splits)
    kf.get_n_splits(X_train)
    final_test_error_list = []
    for train_index, test_index in kf.split(X_train):
        X_train_5fold, X_test_5fold = X_train[train_index], X_train[test_index]
        y_train_5fold, y_test_5fold = y_train[train_index], y_train[test_index]
        
        #### add noises in the same fashion
        X_train_5fold_ori, X_test_5fold_ori = X_train_5fold.copy(), X_test_5fold.copy()
        y_train_5fold_ori, y_test_5fold_ori = y_train_5fold.copy(), y_test_5fold.copy()
        import numpy.matlib
        numOfRepeats = 1
        std_list = [0.5, 1.0, 1.5]
        for i_std, std in enumerate(std_list):
            X_train_5fold = X_train_5fold_ori.copy()
            y_train_5fold = X_train_5fold_ori[:,-y_train_5fold_ori.shape[1]:].copy()
            X_train_5fold = numpy.matlib.repmat(X_train_5fold, numOfRepeats, 1)
            y_train_5fold = numpy.matlib.repmat(y_train_5fold, numOfRepeats, 1)
            noises = np.random.normal(loc=0.0, scale=np.log(10**std), size=y_train_5fold.shape)
            X_train_5fold[:,-y_train_5fold.shape[1]:] = X_train_5fold[:,-y_train_5fold.shape[1]:].copy() + noises
            if i_std==0:
                X_train_5fold_all = X_train_5fold.copy()
                y_train_5fold_all = y_train_5fold.copy()
            else:
                X_train_5fold_all = np.concatenate([X_train_5fold_all, X_train_5fold], axis=0)
                y_train_5fold_all = np.concatenate([y_train_5fold_all, y_train_5fold], axis=0)
        for i_std, std in enumerate(std_list):
            X_train_5fold = X_train_5fold_ori.copy()
            y_train_5fold = X_train_5fold_ori[:,-y_train_5fold_ori.shape[1]:].copy()
            X_train_5fold = numpy.matlib.repmat(X_train_5fold, numOfRepeats, 1)
            y_train_5fold = numpy.matlib.repmat(y_train_5fold, numOfRepeats, 1)
            noises = np.random.uniform(low=-np.sqrt(12)*np.log(10**std)/2, high=np.sqrt(12)*np.log(10**std)/2, size=y_train_5fold.shape)
            X_train_5fold[:,-y_train_5fold.shape[1]:] = X_train_5fold[:,-y_train_5fold.shape[1]:].copy() + noises
            X_train_5fold_all = np.concatenate([X_train_5fold_all, X_train_5fold], axis=0)
            y_train_5fold_all = np.concatenate([y_train_5fold_all, y_train_5fold], axis=0)
        X_train_5fold = X_train_5fold_all.copy()
        y_train_5fold = y_train_5fold_all.copy()

        numOfRepeats = 1
        for i_std, std in enumerate(std_list):
            X_test_5fold = X_test_5fold_ori.copy()
            y_test_5fold = X_test_5fold_ori[:,-y_test_5fold_ori.shape[1]:].copy()
            X_test_5fold = numpy.matlib.repmat(X_test_5fold, numOfRepeats, 1)
            y_test_5fold = numpy.matlib.repmat(y_test_5fold, numOfRepeats, 1)
            noises = np.random.normal(loc=0.0, scale=np.log(10**std), size=y_test_5fold.shape)
            X_test_5fold[:,-y_test_5fold.shape[1]:] = X_test_5fold[:,-y_test_5fold.shape[1]:].copy() + noises
            if i_std==0:
                X_test_5fold_all = X_test_5fold.copy()
                y_test_5fold_all = y_test_5fold.copy()
            else:
                X_test_5fold_all = np.concatenate([X_test_5fold_all, X_test_5fold], axis=0)
                y_test_5fold_all = np.concatenate([y_test_5fold_all, y_test_5fold], axis=0)
        X_test_5fold = X_test_5fold_all.copy()
        y_test_5fold = y_test_5fold_all.copy()
        
        final_test_error = training(X_train_5fold,y_train_5fold,X_test_5fold,y_test_5fold,layer_size,Nlayers,lr,l2,dropout,alpha,beta,batch_size,epochs)
        final_test_error_list = final_test_error_list + [final_test_error]
    return np.nanmean(final_test_error_list)

def prediction(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,layer_size=512,Nlayers=1,lr=0.001,l2=0.001,dropout=0.5,alpha=1.0,beta=1.0,batch_size=64,epochs=50):
    torch.manual_seed(0)
    losses=[]
    X_train, y_train, X_test, y_test= map(
    torch.FloatTensor, (X_train, y_train, X_test, y_test)
    )
    layer_size = int(round(layer_size))
    Nlayers = int(round(Nlayers))

    train_ds = TensorDataset(X_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=batch_size)
    model = Feedforward(hidden_dim=layer_size, Nlayers=Nlayers, dropout=dropout, alpha=alpha, beta=beta)
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=l2)
    test_error = []
    model_list = []
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            pred = model(xb)
            loss = loss_func(pred, yb)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        losses.append(loss)
        model.eval()
        y_pred = model(X_test)
        after_train = spearman(y_pred.squeeze(), y_test) 
        test_error = test_error + [after_train]
        model_list = model_list + [copy.deepcopy(model)]
    if np.where(test_error==np.max(test_error))[0].shape[0] > 0:
        best_model = copy.deepcopy(model_list[np.where(test_error==np.max(test_error))[0][0]])
    else:
        best_model = copy.deepcopy(model)
    best_model.eval()
    y_pred = best_model(X_test)
    mse = mean_squared_error(y_pred.detach().numpy(),y_test.detach().numpy())
    return -mse, y_pred, best_model #to specify what to maximize in this function


from bayes_opt import BayesianOptimization
pbounds = {
    'lr': (1e-5, 1e-5),
    'l2': (1e-9, 1e-9),
    'layer_size': (256, 256),
    'Nlayers': (3, 3), 
    'dropout': (0.0, 0.0),
    'alpha': (0.0, 1.0),
    'beta': (0.0, 0.0)
    }

import random
optimizer = BayesianOptimization(
    f=cross_validation,
    pbounds=pbounds,
    verbose=0, #-1,  
    random_state=42,
)

start = time.time()
optimizer.maximize(init_points=5, n_iter=15)
end = time.time()
print('Bayes optimization takes {:.2f} seconds to tune'.format(end - start))
print(optimizer.max)

l2 = optimizer.max['params']['l2']
lr = optimizer.max['params']['lr']
layer_size = optimizer.max['params']['layer_size']
Nlayers = optimizer.max['params']['Nlayers']
dropout = optimizer.max['params']['dropout']
alpha = optimizer.max['params']['alpha']
beta = optimizer.max['params']['beta']
mse, y_pred, best_model = prediction(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                 layer_size=layer_size,Nlayers=Nlayers,lr=lr,l2=l2,dropout=dropout,alpha=alpha,beta=beta,batch_size=64,epochs=100)

#### evaluate the performance
best_model.eval()
y_pred = best_model(torch.FloatTensor(scaler_x.transform(X_test_ori)))

from scipy.stats import spearmanr
def func_spearman_CC(x_plot, y_plot):
    N_output = y_plot.shape[1]
    spearman_corr = np.zeros(N_output)
    for i in range(N_output):
        spearman_corr[i] = spearmanr(x_plot[:,i], y_plot[:,i])[0]
    return spearman_corr

test_no_biased = y_test_ori.copy()
test_biased = X_test_ori[:,-y_test_ori.shape[1]:]
prediction = scaler_y.inverse_transform(y_pred.detach().numpy())

spearman_corr1 = func_spearman_CC(test_no_biased, test_biased);
spearman_corr2 = func_spearman_CC(test_no_biased, prediction);
spearman_corr3 = func_spearman_CC(test_biased, prediction);

print(np.nanmean(spearman_corr1), np.nanmean(spearman_corr2), np.nanmean(spearman_corr2-spearman_corr1))

########### pickle all processed data which are useful for simulations
import pickle
pickle_out = open(path_prefix_algorithm+"performance.pickle","wb")
pickle.dump([test_no_biased, test_biased, prediction], pickle_out)
pickle_out.close()