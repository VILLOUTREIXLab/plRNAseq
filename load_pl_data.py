import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np 
from sklearn import preprocessing
import torch
import random
import argparse
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.decomposition import PCA
import itertools
from joblib import Parallel, delayed
from joblib import Memory

def load_data_partial_label(PATH, topology, lenght_tree, modules,k, value, overlap, I, t, sub_proportion):
    '''
    return les données train test s, ws avec prior NORMALISEES  en train /test
    '''
    random.seed = 1312
    path_file_abs = PATH

    name_tree = '_'+str(modules)+'_'+str(value)
    path_file = path_file_abs+'data/datasets/final_Tree/' + str(topology) + '/' + str(lenght_tree) + '/'
    dataset = str(topology) + '_' + str(lenght_tree) + '_' + str(modules) + '_' + str(value)
    X = np.load(path_file + 'X_Tree' + str(name_tree) + '.npy', allow_pickle=True)
    y = np.load(path_file + 'y_Tree' + str(name_tree) + '.npy', allow_pickle=True).tolist()
    y = [int(element) for element in y]
    mat_dist = np.load(path_file + 'Tree' + str(name_tree) + '_mat_dist.npy')
    time = np.load(path_file + 'Tree' + str(name_tree) + '_pseudotime.npy')
    # mat_mds = np.load(path_file + str(dataset) + '_mat_mds.npy')
    C = torch.tensor(mat_dist)
    X = np.array(X, 'float')
    c = C.shape[0]
    y_id = torch.eye(c)
    print(dataset)
    #device = 'cuda:0'
    device = 'cpu'
    print(name_tree, X.shape, len(y))


    name_to_save = path_file_abs+'data/datasets/final_Tree/' + str(topology) + '/' + str(lenght_tree)+'/'+str(overlap)+'_'+str(modules)+'_'+str(value)+'_'+str(I)+'_'         
    X_s = np.load(name_to_save+'X_s.npy')
    y_s = np.load(name_to_save+'y_s.npy').tolist()
    y_s_prior = np.load(name_to_save+'y_s_prior.npy').tolist()
    X_ws = np.load(name_to_save+'X_ws.npy')
    y_ws = np.load(name_to_save+'y_ws.npy').tolist()
    y_ws_prior = np.load(name_to_save+'y_ws_prior.npy').tolist()




    if overlap == 0 :

        name_to_save = path_file_abs+'data/datasets/final_Tree/' + str(topology) + '/' + str(lenght_tree)+'/' +str(overlap)+'_'+str(modules)+'_'+str(value)+'_'+str(I)+'_'      
        
        indice_train_s = np.load(name_to_save+'indice_train_s_'+str(t)+'.npy').tolist()
        indice_test_s = np.load(name_to_save+'indice_test_s_'+str(t)+'.npy').tolist()
        indice_train_ws = np.load(name_to_save+'indice_train_ws_'+str(t)+'.npy').tolist()
        indice_test_ws = np.load(name_to_save+'indice_test_ws_'+str(t)+'.npy').tolist()


        X_train_s_0, X_test_s, y_train_0_s, y_test_s, y_train_0_s_prior, y_test_s_prior = torch.FloatTensor(X_s)[indice_train_s], \
            torch.FloatTensor(X_s)[indice_test_s], np.array(y_s)[indice_train_s].tolist(), np.array(y_s)[indice_test_s].tolist(), \
        np.array(y_s_prior)[indice_train_s].tolist(), np.array(y_s_prior)[indice_test_s].tolist()

        X_train_ws_0, X_test_ws, y_train_0_ws, y_test_ws, y_train_0_ws_prior, y_test_ws_prior = torch.FloatTensor(X_ws)[
            indice_train_ws], \
            torch.FloatTensor(X_ws)[indice_test_ws], np.array(y_ws)[indice_train_ws].tolist(), np.array(y_ws)[
            indice_test_ws].tolist(), \
            np.array(y_ws_prior)[indice_train_ws].tolist(), np.array(y_ws_prior)[indice_test_ws].tolist()


        ### choose the amount of supervised data according to n
        sub_indice_train_s, _ = train_test_split(indice_train_s, test_size= 1-sub_proportion, stratify=y_train_0_s)
        X_train_s, y_train_s, y_train_s_prior = torch.FloatTensor(X_s)[sub_indice_train_s], np.array(y_s)[sub_indice_train_s].tolist(), np.array(y_s_prior)[sub_indice_train_s].tolist()
        print('X_train_s_shape : ', X_train_s.shape)

        ## set 200 partial label exemples per label
        sub_indice_train_ws, _ = train_test_split(indice_train_ws, test_size=1-0.2, stratify=y_train_0_ws)
        X_train_ws, y_train_ws, y_train_ws_prior = torch.FloatTensor(X_ws)[sub_indice_train_ws], np.array(y_ws)[sub_indice_train_ws].tolist(), np.array(y_ws_prior)[sub_indice_train_ws].tolist()


        X_test_s, X_test_ws = torch.split(
            torch.FloatTensor(
                preprocessing.StandardScaler().fit_transform(torch.cat((X_test_s, X_test_ws), dim=0))),
            [X_test_s.size(0), X_test_ws.size(0)])


        X_train_s, X_train_ws = torch.split(torch.FloatTensor(preprocessing.StandardScaler().fit_transform(torch.cat((X_train_s, X_train_ws), dim=0))), [X_train_s.size(0), X_train_ws.size(0)])
        y_train_s_prior, y_train_ws_prior = [element[:k] for element in y_train_s_prior], [element[:k] for element in y_train_ws_prior]
        y_test_s_prior, y_test_ws_prior = [element[:k] for element in y_test_s_prior], [element[:k] for element in y_test_ws_prior]

        #print('y_train_ws_prior shape : ',np.shape(np.array(y_train_ws_prior)))

        return C,c,  X_train_s, X_train_ws, y_train_s, y_train_ws, y_train_s_prior, y_train_ws_prior, X_test_s, X_test_ws, y_test_s, y_test_ws, y_test_s_prior, y_test_ws_prior

    if overlap == 1 :

        name_to_save = path_file_abs+'data/datasets/final_Tree/' + str(topology) + '/' + str(lenght_tree)+'/' +str(overlap)+'_'+str(modules)+'_'+str(value)+'_'+str(I)+'_'      
        
        indice_train_s = np.load(name_to_save+'indice_train_s_'+str(t)+'.npy').tolist()
        indice_test_s = np.load(name_to_save+'indice_test_s_'+str(t)+'.npy').tolist()
        indice_train_ws = np.load(name_to_save+'indice_train_ws_'+str(t)+'.npy').tolist()
        indice_test_ws = np.load(name_to_save+'indice_test_ws_'+str(t)+'.npy').tolist()


        X_train_s_0, X_test_s, y_train_0_s, y_test_s, y_train_0_s_prior, y_test_s_prior = torch.FloatTensor(X_s)[indice_train_s], \
                torch.FloatTensor(X_s)[indice_test_s], np.array(y_s)[indice_train_s].tolist(), np.array(y_s)[indice_test_s].tolist(), \
            np.array(y_s_prior)[indice_train_s].tolist(), np.array(y_s_prior)[indice_test_s].tolist()

        X_train_ws_0, X_test_ws, y_train_0_ws, y_test_ws, y_train_0_ws_prior, y_test_ws_prior = torch.FloatTensor(X_ws)[
            indice_train_ws], \
            torch.FloatTensor(X_ws)[indice_test_ws], np.array(y_ws)[indice_train_ws].tolist(), np.array(y_ws)[
            indice_test_ws].tolist(), \
            np.array(y_ws_prior)[indice_train_ws].tolist(), np.array(y_ws_prior)[indice_test_ws].tolist()


        ### choose the amount of supervised data according to n
        test_size_s=0.95
        if sub_proportion == 0.8 :
            test_size_s = 0.5
        sub_indice_train_s, _ = train_test_split(indice_train_s, test_size=test_size_s, stratify=y_train_0_s)
        X_train_s, y_train_s, y_train_s_prior = torch.FloatTensor(X_s)[sub_indice_train_s], np.array(y_s)[sub_indice_train_s].tolist(), np.array(y_s_prior)[sub_indice_train_s].tolist()
        print('X_train_s_shape : ', X_train_s.shape)

        ## set 200 for partial label exemples per labels
        sub_indice_train_ws, _ = train_test_split(indice_train_ws, test_size=0.5, stratify=y_train_0_ws)  ## 200 dans le train
        X_train_ws, y_train_ws, y_train_ws_prior = torch.FloatTensor(X_ws)[sub_indice_train_ws], np.array(y_ws)[sub_indice_train_ws].tolist(), np.array(y_ws_prior)[sub_indice_train_ws].tolist()


        X_test_s, X_test_ws = torch.split(
            torch.FloatTensor(
                preprocessing.StandardScaler().fit_transform(torch.cat((X_test_s, X_test_ws), dim=0))),
            [X_test_s.size(0), X_test_ws.size(0)])


        X_train_s, X_train_ws = torch.split(torch.FloatTensor(preprocessing.StandardScaler().fit_transform(torch.cat((X_train_s, X_train_ws), dim=0))), [X_train_s.size(0), X_train_ws.size(0)])
        y_train_s_prior, y_train_ws_prior = [element[:k] for element in y_train_s_prior], [element[:k] for element in y_train_ws_prior]
        y_test_s_prior, y_test_ws_prior = [element[:k] for element in y_test_s_prior], [element[:k] for element in y_test_ws_prior]

        print('y_train_ws_prior shape : ',np.shape(np.array(y_train_ws_prior)))
        print('proportion vec ', y_train_s.count(0), y_train_ws.count(0))



        return C ,c, X_train_s, X_train_ws, y_train_s, y_train_ws, y_train_s_prior, y_train_ws_prior, X_test_s, X_test_ws, y_test_s, y_test_ws, y_test_s_prior, y_test_ws_prior






def  load_dataset_supervised(PATH, dataset, t, sub_proportion):
    '''
    return les données train test s, ws avec prior NORMALISEES  en train /test
    '''
    random.seed = 1312

    print(dataset)
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
        try :
            path_file = PATH+'/data/datasets/'+str(dataset)+'/'
            X = np.load(path_file+'sample/X.npy', allow_pickle=True)
            y= np.load(path_file+'sample/y.npy', allow_pickle=True).tolist()

            mat_dist = np.load(path_file + str(dataset)+'_mat_dist.npy')
        except :
            path_file = os.getcwd()+'/data/datasets/'+str(dataset)+'/'
            X = np.load(path_file+'sample/X.npy', allow_pickle=True)
            y= np.load(path_file+'sample/y.npy', allow_pickle=True).tolist()

            mat_dist = np.load(path_file + str(dataset)+'_mat_dist.npy')

        if dataset == 'Paul':
            X, y = X[:-1], y[:-1]

    C = torch.Tensor(mat_dist)
    X=np.vstack(X).astype(np.float64)
    X= torch.FloatTensor(X)
    c = C.shape[0]
    print(t, path_file)
    #indice_train, indice_test = np.load(path_file+'sample/'+'_'+str(t)+'_indices_train_.npy', ), np.load(path_file+'sample/' +'_'+str(t)+'_indices_test_.npy', )
    try :
        indice_train, indice_test = np.load(path_file+'sample/'+'_'+str(t)+'_indices_train_.npy', ), np.load(path_file+'sample/' +'_'+str(t)+'_indices_test_.npy', )
        print('train test loaded')
    except :
        print('creation of train/test split and save it')
        X_train_0, X_test, y_train_0, y_test, indice_train, indice_test = train_test_split(X, y, np.arange(X.shape[0])  , test_size=0.165, stratify=y)
        # 1010 train / 200 test pas exemples

        
        print(min([y_train_0.count(e) for e in range(c)]), max([y_train_0.count(e) for e in range(c)]))
        np.save(path_file+'sample/'+'_'+str(t)+'_indices_train_', indice_train)
        np.save(path_file+'sample/'+'_'+str(t)+'_indices_test_', indice_test)


    X_train_0, X_test = X[indice_train], X[indice_test]
    y_train_0, y_test = np.array(y)[indice_train].tolist(), np.array(y)[indice_test].tolist() 

    proportion_vec = [y_train_0.count(e) for e in range(c)]
    print(np.mean(proportion_vec), np.std(proportion_vec), min(proportion_vec), max(proportion_vec))

    #print('SUB PROPORTION' , 1.0001 - sub_proportion ) 
    if sub_proportion == 1.0 :
        sub_indice_train = indice_train.tolist()

    else :
        sub_indice_train, _ = train_test_split(indice_train.tolist(), test_size= 1- sub_proportion, stratify=y_train_0)
    #sub_proportion : 0.98, 0.8 , 0

    X_train_0, y_train_0 = torch.FloatTensor(X)[sub_indice_train], np.array(y)[sub_indice_train].tolist()
     
    print('sub_indice proportion :', sub_proportion)
    proportion_vec = [y_train_0.count(e) for e in range(c)]
    print(np.mean(proportion_vec), np.std(proportion_vec), min(proportion_vec), max(proportion_vec))

    #y_test_prior = [[y] + random.sample(list(set(torch.where(C[y]  < I[1])[0].tolist()) & set(torch.where(C[y]  >= I[0])[0].tolist()) ),  k =k) for y in y_test ]


    return C ,X_train_0, X_test, y_train_0, y_test



def load_dataset_partial_label(PATH, dataset, overlap, I, t, sub_proportion, k, normalize=True):
    '''
    return data train test supervised, partial label, with prior and normalized
    données train test s, ws avec prior NORMALISEES  en train /test
    '''
    random.seed = 1312
    if dataset == 'Packer' : 
        path_file = PATH+'/data/datasets/'+str(dataset)+'/'
        try : 
            X = np.load(path_file+'X_pca.npy', allow_pickle=True)
            y= np.load(path_file+'y.npy', allow_pickle=True).tolist()
            mat_dist = np.load(path_file + str(dataset)+'_mat_dist.npy')
            #print('X,y loaded')
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
        if dataset == 'Paul':  #unique exemple for last label
            X, y = X[:-1], y[:-1]


    C = torch.Tensor(mat_dist)
    X=np.vstack(X).astype(np.float64)
    c = C.shape[0]

    path_file =PATH+'/data/datasets/'+str(dataset)+'/'

    name_to_save = path_file+str(overlap)+'_'+str(I)+'_'


           
    X_s = np.load(name_to_save+'X_s.npy')
    y_s = np.load(name_to_save+'y_s.npy').tolist()
    y_s_prior = np.load(name_to_save+'y_s_prior.npy').tolist()
    X_ws = np.load(name_to_save+'X_ws.npy')
    y_ws = np.load(name_to_save+'y_ws.npy').tolist()
    y_ws_prior = np.load(name_to_save+'y_ws_prior.npy').tolist()

    if overlap == 0 :

        indice_train_s = np.load(name_to_save+'indice_train_s_'+str(t)+'.npy').tolist()
        indice_test_s = np.load(name_to_save+'indice_test_s_'+str(t)+'.npy').tolist()
        indice_train_ws = np.load(name_to_save+'indice_train_ws_'+str(t)+'.npy').tolist()
        indice_test_ws = np.load(name_to_save+'indice_test_ws_'+str(t)+'.npy').tolist()


        X_train_s_0, X_test_s, y_train_0_s, y_test_s, y_train_0_s_prior, y_test_s_prior = torch.FloatTensor(X_s)[indice_train_s], \
            torch.FloatTensor(X_s)[indice_test_s], np.array(y_s)[indice_train_s].tolist(), np.array(y_s)[indice_test_s].tolist(), \
        np.array(y_s_prior)[indice_train_s].tolist(), np.array(y_s_prior)[indice_test_s].tolist()

        X_train_ws_0, X_test_ws, y_train_0_ws, y_test_ws, y_train_0_ws_prior, y_test_ws_prior = torch.FloatTensor(X_ws)[
            indice_train_ws], \
            torch.FloatTensor(X_ws)[indice_test_ws], np.array(y_ws)[indice_train_ws].tolist(), np.array(y_ws)[
            indice_test_ws].tolist(), \
            np.array(y_ws_prior)[indice_train_ws].tolist(), np.array(y_ws_prior)[indice_test_ws].tolist()


        ### amount of supervised data p
        if sub_proportion == 1.0 :
            sub_indice_train_s = indice_train_s
        else :
            try :
                sub_indice_train_s, _ = train_test_split(indice_train_s, test_size= 1-sub_proportion, stratify=y_train_0_s)
            except :
                sub_indice_train_s, _ = train_test_split(indice_train_s, test_size= 1-sub_proportion)
        X_train_s, y_train_s, y_train_s_prior = torch.FloatTensor(X_s)[sub_indice_train_s], np.array(y_s)[sub_indice_train_s].tolist(), np.array(y_s_prior)[sub_indice_train_s].tolist()
        #print('X_train_s_shape : ', X_train_s.shape)


        X_train_ws, y_train_ws, y_train_ws_prior = torch.FloatTensor(X_ws), np.array(y_ws).tolist(), np.array(y_ws_prior).tolist()
        if normalize :
            X_train_s, X_train_ws = torch.split(torch.FloatTensor(preprocessing.StandardScaler().fit_transform(torch.cat((X_train_s, X_train_ws), dim=0))),
                [X_train_s.size(0), X_train_ws.size(0)])

        X_test_s, X_test_ws = torch.split(torch.FloatTensor(preprocessing.StandardScaler().fit_transform(torch.cat((X_test_s, X_test_ws), dim=0))), [X_test_s.size(0), X_test_ws.size(0)])

        
        y_train_s_prior, y_train_ws_prior = [element[:k] for element in y_train_s_prior], [element[:k] for element in y_train_ws_prior]
        y_test_s_prior, y_test_ws_prior = [element[:k] for element in y_test_s_prior], [element[:k] for element in y_test_ws_prior]


        return C,c,  X_train_s, X_train_ws, y_train_s, y_train_ws, y_train_s_prior, y_train_ws_prior, X_test_s, X_test_ws, y_test_s, y_test_ws, y_test_s_prior, y_test_ws_prior

    if overlap == 1 :

        indice_train_s = np.load(name_to_save+'indice_train_s_'+str(t)+'.npy').tolist()
        indice_test_s = np.load(name_to_save+'indice_test_s_'+str(t)+'.npy').tolist()
        indice_train_ws = np.load(name_to_save+'indice_train_ws_'+str(t)+'.npy').tolist()
        indice_test_ws = np.load(name_to_save+'indice_test_ws_'+str(t)+'.npy').tolist()


        X_train_s_0, X_test_s, y_train_0_s, y_test_s, y_train_0_s_prior, y_test_s_prior = torch.FloatTensor(X_s)[indice_train_s], \
                torch.FloatTensor(X_s)[indice_test_s], np.array(y_s)[indice_train_s].tolist(), np.array(y_s)[indice_test_s].tolist(), \
            np.array(y_s_prior)[indice_train_s].tolist(), np.array(y_s_prior)[indice_test_s].tolist()

        X_train_ws_0, X_test_ws, y_train_0_ws, y_test_ws, y_train_0_ws_prior, y_test_ws_prior = torch.FloatTensor(X_ws)[
            indice_train_ws], \
            torch.FloatTensor(X_ws)[indice_test_ws], np.array(y_ws)[indice_train_ws].tolist(), np.array(y_ws)[
            indice_test_ws].tolist(), \
            np.array(y_ws_prior)[indice_train_ws].tolist(), np.array(y_ws_prior)[indice_test_ws].tolist()


        test_size_s=0.8
        if sub_proportion == 0.8 :
            test_size_s = 0.5

        if sub_proportion == 1.0 :
            sub_indice_train_s = indice_train_s
        else : 
            try :
                sub_indice_train_s, _ = train_test_split(indice_train_s, test_size=1 - sub_proportion, stratify=y_train_0_s)
            except : 
                sub_indice_train_s, _ = train_test_split(indice_train_s, test_size= 1-sub_proportion)
        X_train_s, y_train_s, y_train_s_prior = torch.FloatTensor(X_s)[sub_indice_train_s], np.array(y_s)[sub_indice_train_s].tolist(), np.array(y_s_prior)[sub_indice_train_s].tolist()

        X_train_ws, y_train_ws, y_train_ws_prior = torch.FloatTensor(X_ws), np.array(y_ws).tolist(), np.array(y_ws_prior).tolist()


        #normalise train data
        if normalize :
            X_train_s, X_train_ws = torch.split(torch.FloatTensor(preprocessing.StandardScaler().fit_transform(torch.cat((X_train_s, X_train_ws), dim=0))), [X_train_s.size(0), X_train_ws.size(0)])

        # normalize test data
        X_test_s, X_test_ws = torch.split(torch.FloatTensor(preprocessing.StandardScaler().fit_transform(torch.cat((X_test_s, X_test_ws), dim=0))),  [X_test_s.size(0), X_test_ws.size(0)])

        y_train_s_prior, y_train_ws_prior = [element[:k] for element in y_train_s_prior], [element[:k] for element in y_train_ws_prior]
        y_test_s_prior, y_test_ws_prior = [element[:k] for element in y_test_s_prior], [element[:k] for element in y_test_ws_prior]

        return C ,c, X_train_s, X_train_ws, y_train_s, y_train_ws, y_train_s_prior, y_train_ws_prior, X_test_s, X_test_ws, y_test_s, y_test_ws, y_test_s_prior, y_test_ws_prior


