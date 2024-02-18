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
# from pl_model import pl_SVM, pl_nn_RIC, pl_KNN, pl_ric_light, pl_ultra_light_ric
from load_pl_data import load_dataset_partial_label
from pl_model import pl_hKNN, plSVM, pl_nn_prototybe_based


from init_dict import init_dict_0, init_method



# from pl_setting import split_partial_label


#### You may need parallelization computational ressourrces. CPU is large enough, but 10 or 20 cpus can be great
# choose a multiple
# %%
developpement = False  # debugging
device = 'cpu'  # device = 'cuda:0'
PATH = os.getcwd()
path_file_abs = PATH
random.seed= 1312
print_parameter = False
# %%
dataset_liste = ['Packer', # C. Elegans
                    'Paul',  # Myeloid Progenitor
                     'Planaria'  # S. Mediterranea



                     ]
#COMING SOON [, 'linear', 'half', 'binary']
dataset = dataset_liste[2]







# PARTIAL LABELLING SETTING
overlap = 1
p = 0.1
k = 2
I = 'I0'









# METHOD
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

method = method_liste[0]
linear = False  # False #neural network for PB or kernel for SVM
indice_network = 2 if linear == False else 0

# HIERARCHY

choix_C_liste = ['flat', 'C']
choix_C = choix_C_liste[0]

# %%

if dataset in ['linear', 'half', 'binary']:
    alpha = 0.1  # 0.5
    programs = 50
    lenght_tree = 8
    topology = 'half'
    nfactor = 5
    p = 200

    name_tree = '_' + str(programs) + '_' + str(alpha) + '_nfactor_' + str(nfactor)
    path_file = path_file_abs + 'data/datasets/Tree/' + str(topology) + '/' + str(lenght_tree) + '/'
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

if dataset in ['Packer', 'Paul', 'Planaria']:

    if dataset == 'Packer':
        path_file = PATH + '/data/datasets/' + str(dataset) + '/'
        try:
            X = np.load(path_file + 'X_pca.npy', allow_pickle=True)
            y = np.load(path_file + 'y.npy', allow_pickle=True).tolist()
            mat_dist = np.load(path_file + str(dataset) + '_mat_dist.npy')
            #print('X,y loaded')
        except:

            df = pd.read_csv(path_file + dataset + '.csv')
            X = np.asarray(df)
            X = X[:, :-1]
            np.save(path_file + 'X', X)
            try:
                y_names = df['labels'].tolist()
            except:
                y_names = df['cell_type'].tolist()
            names = np.load(path_file + dataset + '/' + dataset + '_names.npy').tolist()
            names = sorted(list(set(names)))
            y = [names.index(i) for i in y_names]
            np.save(path_file + 'y', y)
            mat_dist = np.load(path_file + str(dataset) + '_mat_dist.npy')

    if dataset == 'Planaria' or dataset == 'Paul':
        #
        path_file = PATH + '/data/datasets/' + str(dataset) + '/'
        X = np.load(path_file + 'sample/X.npy', allow_pickle=True)
        y = np.load(path_file + 'sample/y.npy', allow_pickle=True).tolist()

        mat_dist = np.load(path_file + str(dataset) + '_mat_dist.npy')
        # except :
        #     path_file = os.getcwd()+'/data/datasets/'+str(dataset)+'/'
        #     X = np.load(path_file+'sample/X.npy', allow_pickle=True)
        #     y= np.load(path_file+'sample/y.npy', allow_pickle=True).tolist()

        #     mat_dist = np.load(path_file + str(dataset)+'_mat_dist.npy')

        if dataset == 'Paul':
            X, y = X[:-1], y[:-1]

    C = torch.Tensor(mat_dist)
    X = np.vstack(X).astype(np.float64)
    print(X.shape, len(y))
    # X= torch.FloatTensor(X)
    c = C.shape[0]

mat_C = C if choix_C == 'C' else torch.ones((c, c)) - torch.eye(c)

# %%
C, c, X_train_s, X_train_ws, y_train_s, y_train_ws, \
    y_train_s_prior, y_train_ws_prior, X_test_s, X_test_ws, \
    y_test_s, y_test_ws, y_test_s_prior, y_test_ws_prior \
    = load_dataset_partial_label(PATH=PATH,
                                 dataset=dataset,
                                 overlap=overlap,
                                 I=I,
                                 t=0,
                                 sub_proportion=p,
                                 k=k)


# %%



# %%
dict_0, model = init_dict_0(method=method, indice_network=indice_network, dim_int=X_train_s.size(1), c=c, mat_C=mat_C, C=C, device=device,  developpement=developpement)


# model = init_method(method, dict_0)
if method == 'plhKNN' and choix_C == 'flat':
    model.flat = True

print('dataset :', dataset, ', overlap : ', overlap, ', k : ', k, '  p : ', p,' I : ', I,)
print('Method : ', method, ' linear : ', linear, ' hierarchy : ', choix_C)



#%%
all_results_t = []
for t in range(5):

    C, c, X_train_s, X_train_ws, y_train_s, y_train_ws, \
        y_train_s_prior, y_train_ws_prior, X_test_s, X_test_ws, \
        y_test_s, y_test_ws, y_test_s_prior, y_test_ws_prior \
        = load_dataset_partial_label(PATH=PATH,
                                     dataset=dataset,
                                     overlap=overlap,
                                     I=I,
                                     t=t,
                                     sub_proportion=p,
                                     k=k,
                                     normalize = False)


    #print('FIT')
    dict_0, model = init_dict_0(method=method, indice_network=indice_network, dim_int=X_train_s.size(1), c=c, mat_C=mat_C, C=C, device=device,  developpement=developpement)
    #model = init_method(method, dic_0)
    if method == 'plhKNN' and choix_C == 'flat':
        model.flat = True

    record = []

    for v in range(5):

        try:
            X_train_s_v, X_val_s_v, y_train_s_v, y_val_s_v, y_train_s_prior_v, y_val_s_prior_v = train_test_split(
                X_train_s, y_train_s, y_train_s_prior, test_size=0.2, stratify=y_train_s)
            X_train_ws_v, X_val_ws_v, y_train_ws_v, y_val_ws_v, y_train_ws_prior_v, y_val_ws_prior_v = train_test_split(
                X_train_ws, y_train_ws, y_train_ws_prior, test_size=0.2, stratify=y_train_ws)
        except:
            # NO STRATIFY
            X_train_s_v, X_val_s_v, y_train_s_v, y_val_s_v, y_train_s_prior_v, y_val_s_prior_v = train_test_split(
                X_train_s, y_train_s, y_train_s_prior, test_size=0.2)
            X_train_ws_v, X_val_ws_v, y_train_ws_v, y_val_ws_v, y_train_ws_prior_v, y_val_ws_prior_v = train_test_split(
                X_train_ws, y_train_ws, y_train_ws_prior, test_size=0.2)

        X_val_s_v, X_val_ws_v = torch.split(torch.FloatTensor(
            preprocessing.StandardScaler().fit_transform(torch.cat((X_val_s_v, X_val_ws_v), dim=0))),
                                            [X_val_s_v.size(0), X_val_ws_v.size(0)])
        X_train_s_v, X_train_ws_v = torch.split(torch.FloatTensor(
            preprocessing.StandardScaler().fit_transform(torch.cat((X_train_s_v, X_train_ws_v), dim=0))),
                                                [X_train_s_v.size(0), X_train_ws_v.size(0)])


        #### ATTENTION LES DONNES VAL ET TRAIN sont normalisées ensemble, pas 100% propres

        def gridsearch(dict_entry):

   

            model = init_method(method, dict_entry)

            model.fit(X_train=torch.cat((X_train_s_v, X_train_ws_v), dim=0),
                      y_train=[[yi] for yi in y_train_s_v] + y_train_ws_prior_v)

            setting = [model, str(dataset) + '_' + str(overlap), k, I, overlap]

            pred_train_s_v = model.predict(X=X_train_s_v, y_prior=None)
            pred_train_s_prior_v = model.predict(X=X_train_s_v, y_prior=y_train_s_prior_v)
            pred_train_ws_v = model.predict(X=X_train_ws_v, y_prior=None)
            pred_train_ws_prior_s = model.predict(X=X_train_ws_v, y_prior=y_train_ws_prior_v)

            pred_val_s_v = model.predict(X=X_val_s_v, y_prior=None)
            pred_val_s_prior_v = model.predict(X=X_val_s_v, y_prior=y_val_s_prior_v)
            pred_val_ws_v = model.predict(X=X_val_ws_v, y_prior=None)
            pred_val_ws_prior_s = model.predict(X=X_val_ws_v, y_prior=y_val_ws_prior_v)

            performance = [model.score(y_test=y_train_s_v, y_pred=pred_train_s_v),
                           model.score(y_test=y_train_s_v, y_pred=pred_train_s_prior_v),
                           model.score(y_test=y_train_ws_v, y_pred=pred_train_ws_v),
                           model.score(y_test=y_train_ws_v, y_pred=pred_train_ws_prior_s),
                           model.score(y_test=y_val_s_v, y_pred=pred_val_s_v),
                           model.score(y_test=y_val_s_v, y_pred=pred_val_s_prior_v),
                           model.score(y_test=y_val_ws_v, y_pred=pred_val_ws_v),
                           model.score(y_test=y_val_ws_v, y_pred=pred_val_ws_prior_s),

                           -model.error(y_test=y_train_s_v, y_pred=pred_train_s_v, C=model.C_score),
                           -model.error(y_test=y_train_s_v, y_pred=pred_train_s_prior_v, C=model.C_score),
                           -model.error(y_test=y_train_ws_v, y_pred=pred_train_ws_v, C=model.C_score),
                           -model.error(y_test=y_train_ws_v, y_pred=pred_train_ws_prior_s, C=model.C_score),
                           -model.error(y_test=y_val_s_v, y_pred=pred_val_s_v, C=model.C_score),
                           -model.error(y_test=y_val_s_v, y_pred=pred_val_s_prior_v, C=model.C_score),
                           -model.error(y_test=y_val_ws_v, y_pred=pred_val_ws_v, C=model.C_score),
                           -model.error(y_test=y_val_ws_v, y_pred=pred_val_ws_prior_s, C=model.C_score),

                           ]

            # return dict_entry, setting, performance
            return performance


        dict_0, model = init_dict_0(method=method, indice_network=indice_network, dim_int=X_train_s.size(1), c=c, mat_C=mat_C, C=C, device=device,  developpement=developpement)



        grid = model.param_grid()
        if developpement:
            grid = model.tiny_param_grid()

        keys, values = zip(*grid.items())
        gridsearch_parameter = [dict(zip(keys, v)) for v in itertools.product(*values)]

        dict_gridsearch_parameter = gridsearch_parameter
        for element in range(len(dict_gridsearch_parameter)):
            for (keys, values) in dict_0.items():
                dict_gridsearch_parameter[element][keys] = values

        print('val :', v, 'grisearch size :', len(dict_gridsearch_parameter))

        all_result_v = Parallel(n_jobs=-1)(
            delayed(gridsearch)(combinaison) for combinaison in dict_gridsearch_parameter)
        record.append(all_result_v)



    array_result = np.array(record)
    #print(array_result.shape)

    results_perf = np.array(array_result)
    result_mean = np.mean(results_perf, axis=0)

    best_hyperparams_combi_index = np.argmax(np.mean(results_perf, axis=0), axis=0).tolist()
    best_hyperparams_combi = [gridsearch_parameter[element] for element in best_hyperparams_combi_index]



    def gridsearch_test(i):
        dict_entry = best_hyperparams_combi[i]
        model = init_method(method, dict_entry)

        model.fit(X_train=torch.cat((X_train_s, X_train_ws), dim=0),
                  y_train=[[yi] for yi in y_train_s] + y_train_ws_prior)

        if method == 'plhKNN' and choix_C == 'C_flat':
            model.flat = True

        setting = [model, str(dataset) + '_' + str(overlap), k, I, overlap]

        pred_train_s = model.predict(X=X_train_s, y_prior=None)
        pred_train_s_prior = model.predict(X=X_train_s, y_prior=y_train_s_prior)
        pred_train_ws = model.predict(X=X_train_ws, y_prior=None)
        pred_train_ws_prior_s = model.predict(X=X_train_ws, y_prior=y_train_ws_prior)

        pred_test_s = model.predict(X=X_test_s, y_prior=None)
        pred_test_s_prior = model.predict(X=X_test_s, y_prior=y_test_s_prior)
        pred_test_ws = model.predict(X=X_test_ws, y_prior=None)
        pred_test_ws_prior_s = model.predict(X=X_test_ws, y_prior=y_test_ws_prior)

        performance = [model.score(y_test=y_train_s, y_pred=pred_train_s),
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

        # return dict_entry, setting, performance
        return performance


    test_result = Parallel(n_jobs=-1)(
        delayed(gridsearch_test)(combinaison) for combinaison in range(len(best_hyperparams_combi)))

    print('TEST : ', t)  # , test_result)
    # print(test_result)
    all_results_t.append(test_result)
    # record.append(all_result_v[-1][-1])


array_result = np.array(all_results_t)

results_perf = np.array(array_result)
#print(results_perf.shape)

best_results_t = np.array([results_perf[:, i, i] for i in range(len(best_hyperparams_combi))])
best_results = np.mean(best_results_t, axis=1)





print('dataset :', dataset, ', overlap : ', overlap, ', k : ', k, '  p : ', p,' I : ', I,)
print('Method : ', method, ' linear : ', linear, ' hierarchy : ', choix_C)
print(' Test set :   Supervised,       PL,      PL Prior')
print('AVERAGE RESULTS : ', np.mean(best_results_t, axis=1)[[4,6,7]].tolist())
print('STD       : ', np.std(best_results_t, axis=1)[[4,6,7]].tolist())


### For check best params on average validation set for t fixed:
# print(best_hyperparams_combi[CRIT]
if print_parameter :
    print(best_hyperparams_combi[4], best_hyperparams_combi[6], best_hyperparams_combi[7])