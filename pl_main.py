import os
import pandas as pd
import torch
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import random
import argparse
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.decomposition import PCA
import itertools
from joblib import Parallel, delayed
from joblib import Memory
from sklearn.neighbors import KNeighborsClassifier
import time
from collections import Counter



from load_pl_data import load_dataset_partial_label



from init_dict import init_dict_0, init_method


#from pl_setting import split_partial_label

#%%
developpement = False # debugging
device = 'cpu'    #device = 'cuda:0'
PATH = os.getcwd()
path_file_abs = PATH
random.seed= 1312
#%%

dataset_liste = ['Packer', # C. Elegans
                    'Paul',  # Myeloid Progenitor
                     'Planaria',  # S. Mediterranea


                     'linear','half','binary']
dataset = dataset_liste[2]

# PARTIAL LABELLING SETTING 
overlap = 1  #[0,1]
p = 0.1 # [0.1, 1.0] in the paper
k=2 #[2,4,10]
I = 'I0'
method_liste = [

# ALGO IRL 
'PB',  # Prototype Based
'plSVM',  # Support Vector Machine  
'plLR',  # Logistic Regression

# Special

'plhKNN',  # k Nearest Neighbors

# Algo IFR 
'plRF', # Random Forest
'plXGBM', # Extreme Gradient Boosting Method
'plkernelSVC', # kernel SVM (real kernel implmentation)
'plEMLR',   # LR with algo IFR
'plEMSVM', # SVM with algo IFR
]

method = method_liste[1]
linear = False #False #neural network for PB or kernel for SVM
indice_network = 2 if linear == False  else 0

#HIERARCHY

choix_C_liste = ['flat','C']
choix_C = choix_C_liste[0]

#%%

if dataset in ['linear','half','binary'] :
    alpha = 0.1  #0.5
    programs = 50
    lenght_tree = 8
    topology = 'half'
    nfactor=5
    p=200


    name_tree = '_'+str(programs)+'_'+str(alpha)+'_nfactor_'+str(nfactor)
    path_file = path_file_abs+'data/datasets/Tree/' + str(topology) + '/' + str(lenght_tree) + '/'
    dataset = str(topology) + '_' + str(lenght_tree) + '_' + str(programs) + '_' + str(alpha)


    X = np.load(path_file + 'X_Tree' + str(name_tree) + '.npy', allow_pickle=True)
    y = np.load(path_file + 'y_Tree' + str(name_tree) + '.npy', allow_pickle=True).tolist()
    y = [int(element) for element in y]
    mat_dist = np.load(path_file + 'Tree' + str(name_tree) + '_mat_dist.npy')
    time = np.load(path_file + 'Tree' + str(name_tree) + '_pseudotime.npy')

    C = torch.tensor(mat_dist)
    X = np.array(X, 'float')
    c = C.shape[0]
    y_id = torch.eye(c)
    print(dataset)

    
    print(name_tree, X.shape, len(y))

if dataset in ['Packer', 'Paul','Planaria'] :

    if dataset == 'Packer' : 
            path_file = PATH+'/data/datasets/'+str(dataset)+'/'
            try : 
                X = np.load(path_file+'X_pca.npy', allow_pickle=True)
                y= np.load(path_file+'y.npy', allow_pickle=True).tolist()
                mat_dist = np.load(path_file + str(dataset)+'_mat_dist.npy')
                print('X,y loaded')
            except :

                df = pd.read_csv(path_file + dataset + '.csv')
                X = np.asarray(df)
                X = X[:, :-1]
                np.save(path_file  +'X', X)
                try:
                    y_names = df['labels'].tolist()
                except:
                    y_names = df['cell_type'].tolist()
                names = np.load(path_file + dataset + '/' + dataset + '_names.npy').tolist()
                names = sorted(list(set(names)))
                y = [names.index(i) for i in y_names]
                np.save(path_file+'y', y)
                mat_dist = np.load(path_file + str(dataset)+'_mat_dist.npy')

    if dataset == 'Planaria' or dataset == 'Paul' :
        #
        path_file = PATH+'/data/datasets/'+str(dataset)+'/'
        X = np.load(path_file+'sample/X.npy', allow_pickle=True)
        y= np.load(path_file+'sample/y.npy', allow_pickle=True).tolist()

        mat_dist = np.load(path_file + str(dataset)+'_mat_dist.npy')
        # except :
        #     path_file = os.getcwd()+'/data/datasets/'+str(dataset)+'/'
        #     X = np.load(path_file+'sample/X.npy', allow_pickle=True)
        #     y= np.load(path_file+'sample/y.npy', allow_pickle=True).tolist()

        #     mat_dist = np.load(path_file + str(dataset)+'_mat_dist.npy')

        if dataset == 'Paul':
            X, y = X[:-1], y[:-1]


    C = torch.Tensor(mat_dist)
    X=np.vstack(X).astype(np.float64)
    print(X.shape, len(y))
    #X= torch.FloatTensor(X)
    c = C.shape[0]



mat_C = C if choix_C == 'C' else torch.ones((c,c))- torch.eye(c)


#%%
C ,c, X_train_s, X_train_ws, y_train_s, y_train_ws,\
    y_train_s_prior, y_train_ws_prior, X_test_s, X_test_ws, \
    y_test_s, y_test_ws, y_test_s_prior, y_test_ws_prior \
    = load_dataset_partial_label(PATH=PATH,
                                 dataset=dataset,
                                 overlap=overlap,
                                 I=I,
                                 t=0,
                                 sub_proportion=p,
                                 k=k)


#%%
dict_0, model = init_dict_0(method=method, indice_network=indice_network, dim_int=X_train_s.size(1), c=c, mat_C=mat_C, C=C, device=device,  developpement=developpement)

grid = model.tiny_param_grid()  #  spectific parameters which require gridsearch

for (keys, values) in grid.items():
    dict_0[keys] = random.choice(values)


model = init_method(method=method, dict_entry=dict_0)




if method == 'plhKNN' and choix_C == 'flat':
    model.flat = True
#%%
print('FIT')
model.fit(X_train=torch.cat((X_train_s, X_train_ws), dim=0),
                    y_train= [[yi] for yi in y_train_s] + y_train_ws_prior)

print('PRED TEST')
pred_train_s = model.predict(X=X_train_s, y_prior=None)
pred_train_s_prior = model.predict(X=X_train_s, y_prior=y_train_s_prior)
pred_train_ws = model.predict(X=X_train_ws, y_prior=None)
pred_train_ws_prior_s = model.predict(X=X_train_ws, y_prior=y_train_ws_prior)

pred_test_s = model.predict(X=X_test_s, y_prior=None)
pred_test_s_prior = model.predict(X=X_test_s, y_prior=y_test_s_prior)
pred_test_ws = model.predict(X=X_test_ws, y_prior=None)
pred_test_ws_prior_s = model.predict(X=X_test_ws, y_prior=y_test_ws_prior)


performance = [ model.score(y_test=y_train_s, y_pred=pred_train_s),
                model.score(y_test=y_train_s, y_pred=pred_train_s_prior),
                model.score(y_test=y_train_ws, y_pred=pred_train_ws),
                model.score(y_test=y_train_ws, y_pred=pred_train_ws_prior_s),
                model.score(y_test=y_test_s, y_pred=pred_test_s),
                model.score(y_test=y_test_s, y_pred=pred_test_s_prior),
                model.score(y_test=y_test_ws, y_pred=pred_test_ws),
                model.score(y_test=y_test_ws, y_pred=pred_test_ws_prior_s),


                model.error(y_test=y_train_s, y_pred=pred_train_s, C=model.C_score),
                model.error(y_test=y_train_s, y_pred=pred_train_s_prior, C=model.C_score),
                model.error(y_test=y_train_ws, y_pred=pred_train_ws, C=model.C_score),
                model.error(y_test=y_train_ws, y_pred=pred_train_ws_prior_s, C=model.C_score),
                model.error(y_test=y_test_s, y_pred=pred_test_s, C=model.C_score),
                model.error(y_test=y_test_s, y_pred=pred_test_s_prior, C=model.C_score),
                model.error(y_test=y_test_ws, y_pred=pred_test_ws, C=model.C_score),
                model.error(y_test=y_test_ws, y_pred=pred_test_ws_prior_s, C=model.C_score),


                ]

print('dataset :', dataset, ', overlap : ', overlap, ', k : ', k, '  p : ', p,' I : ', I,)
print('Method : ', method, ' linear : ', linear, ' hierarchy : ', choix_C)
print('supervised test set accuracy : ',performance[4])
print('partial label test set accuracy : ',performance[6])
print('partial label test set accuracy with prior : ',performance[7])