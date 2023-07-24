import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torchmetrics.functional import pairwise_cosine_similarity


import random
import argparse
import itertools
from joblib import Parallel, delayed, Memory
import time
from collections import Counter

#%%

from sklearn import preprocessing



from sklearn.manifold import MDS
from sklearn.manifold import Isomap

from sklearn.metrics import make_scorer
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import f1_score

from sklearn.kernel_approximation import RBFSampler






######################## Partial Label model #####################"""""""




class pl_hKNN():
    def __init__(self, k=10, C=[], C_score=[], flat=False):
        self.k = k
        self.C = C
        self.C_score = C_score
        self.flat = flat

    def fit(self, X_train, y_train, liste_y=[], **dict_params):
        self.X_train = X_train
        self.y_train = self.reshape_liste_y(y_train)
        self.liste_y_train = liste_y
        self.fully_supervised = False

    def reshape_liste_y(self, liste_y):
        lenght_max = max([len(element) for element in liste_y])
        y_train_shape_max = [
            element if len(element) == lenght_max else element + [element[-1]] * (lenght_max - len(element)) for element
            in liste_y]

        return y_train_shape_max

    def predict(self, X, y_prior=None):
        batch_size = 2000

        result = []

        for i in range(0, X.size()[0], batch_size):


            batch_Z = (X[i:i + batch_size])

            pairwisedist = torch.cdist(self.X_train, batch_Z).to('cpu')

            topk = torch.topk(pairwisedist.T, largest=False, k=self.k).indices  # indices des kplus proches voisins

            ytopk = torch.tensor(self.y_train)[topk]
            ytopk = ytopk.reshape(ytopk.shape[0], -1)

            if self.flat :
                result = result + torch.mode(ytopk, dim=1)[0].detach().tolist()
            else : 

                all_votes = (1 / 2 ** (self.C[:, ytopk])).sum(dim=2)

                if y_prior is None :
                    argmax_votes = torch.argmax(all_votes, dim=0)
                    result = result + argmax_votes.tolist()

                if y_prior is not None :

                    batch_prior = torch.tensor(y_prior)[i:i + batch_size]
                    all_votes_prior = torch.tensor([(all_votes[i, ix]).tolist() for ix, i in enumerate(batch_prior)])
                    argmax_prior_votes = torch.argmax(all_votes_prior, dim=1)
                    liste_result = [(batch_prior[i][argmax_prior_votes[i]]).tolist() for i in
                                    range(argmax_prior_votes.shape[0])]
                    result = result + liste_result

        return result

    def score(self, y_test ,X_test=None , y_prior=None, y_pred=None):
        if y_pred is None :
            y_pred = self.predict(X_test)
        return f1_score(y_pred, y_test, average='micro')

    def error(self, y_test, y_pred=None, X_test=None, y_prior=None, C=torch.ones((253, 253))):
        if y_pred is None:
            y_pred = self.predict(X_test, y_prior=y_prior)

        vec_C_error = C[torch.tensor(y_pred).reshape(-1,1), torch.tensor(y_test).reshape(-1,1)]
        return (vec_C_error.sum()/torch.count_nonzero(vec_C_error)).detach().tolist()
    

    def param_grid(self):
        param_grid = {"k": [1,10, 20, 50, 100],  # 5
                      }
        return param_grid

    def tiny_param_grid(self):
        param_grid = {"k": [2, 10, 50],  # 5
                      }
        return param_grid

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self






class pl_nn_prototybe_based():

    def __init__(self, Network=nn.Sequential(nn.Linear(26,26)) ,
                     lambdaa=1e-5,    #regularisation
                     lambdaa_1=1e-5,
                     lambdaa_solution_regression=1.e-5 ,  #regularisation sur la solution de la regression
                     optimizer = 'Adam',
                     lr_P = 1.e-5, 
                     y_ref = 'id',

                     epochs_regression = 101,
                     epochs_xsi=101, 
                      
                     distance = 'correlation',  #euclidian
                     device='cpu', 
                     C=torch.ones(26),
                     C_score=torch.ones(26)):

        self.device = device

        self.lambdaa = lambdaa
        self.lambdaa_1 = lambdaa_1
        self.max_epoch_regression = epochs_regression
        self.max_epoch_xsi = epochs_xsi
        self.C = C
        self.C_score = C_score
        self.c = C.shape[0]
        self.lr_P = lr_P
        self.Network = Network.to(self.device)
         # Sequentiel
        if optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(params=self.Network.parameters(), weight_decay=self.lambdaa)
            self.optimizer_xsi = torch.optim.Adam(params=self.Network.parameters(), weight_decay=self.lambdaa_1)


        if optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(params=self.Network.parameters(), weight_decay=self.lambdaa, lr=self.lr_P)
            self.optimizer_xsi = torch.optim.SGD(params=self.Network.parameters(), weight_decay=self.lambdaa_1, lr=self.lr_P)

        self.distance = distance
        if distance == 'euclidian':
            self.pairwisedist = torch.cdist

        if distance == 'correlation':
            
            self.pairwisedist = lambda x0,x1 : 1- pairwise_cosine_similarity(x0,x1)

        self.lambdaa_solution_regression = lambdaa_solution_regression


        self.yref = torch.eye(self.c).to(self.device)
        if y_ref == 'mds' :


            self.d = 100
            embedding = MDS(n_components=self.d, dissimilarity='precomputed')
            self.y_ref = torch.Tensor(embedding.fit_transform(self.C)).to(self.device)

        self.C = self.C.to(self.device)




    def optimize_regression(self, X, liste_y, C, batch_size=200, save_loss=False):

        permutation = torch.randperm(X.size()[0])
        L_tot = 0
        for i in range(0, X.size()[0], batch_size):
            indices = permutation[i:i + batch_size]
            batch_Z = (X[indices]).to(self.device)
            batch_y = [liste_y[indice_y] for indice_y in indices]

            
            pairwise_dist = self.pairwisedist(self.Network(batch_Z), self.yref)
            pairwist_dist_pl = pairwise_dist.gather(1, torch.tensor(batch_y).to(self.device)).topk(k=1, largest=False)[0]

            all_xsi_i_alpha = pairwist_dist_pl 
            loss = torch.max(all_xsi_i_alpha, torch.zeros(1).to(self.device)).sum()
            #print(loss)
            
            self.optimizer.zero_grad()
            
            loss.backward()
            self.optimizer.step()
            
            with torch.no_grad():
                L_tot += loss

        if save_loss :
            pred_train = torch.argmin(self.pairwisedist(self.Network(X.to(self.device)), self.yref), dim=1).tolist()
            return L_tot, pred_train

    def optimize_xsi(self, X, liste_y, C, batch_size=200, save_loss=False):

        permutation = torch.randperm(X.size()[0])
        L_tot = 0
        for i in range(0, X.size()[0], batch_size):
            indices = permutation[i:i + batch_size]
            batch_Z = (X[indices]).to(self.device)
            batch_y = [liste_y[indice_y] for indice_y in indices]

            
            

            pairwise_dist = self.pairwisedist(self.Network(batch_Z),self.yref)
            pairwist_dist_pl = pairwise_dist.gather(1, torch.tensor(batch_y).to(self.device)).topk(k=1, largest=False)[0]

            index = (torch.tensor(batch_y).to(self.device)).gather(1, pairwise_dist.gather(1, torch.tensor(batch_y).to(self.device)).topk(k=1, largest=False)[1])
  
            all_xsi_i_alpha = pairwist_dist_pl - pairwise_dist + self.C[index, :].reshape(batch_Z.shape[0],  self.c)
            


            l2_reg = torch.zeros(1).to(self.device)
            for i, params in enumerate(self.Network.parameters()):
                l2_reg += torch.norm(self.param_freeze[i] - params)



            #regularisation_solution = self.lambdaa_solution_regression * torch.norm( self.Network(batch_Z) -      self.solution_regression[indices])
            


            loss = torch.max(all_xsi_i_alpha, torch.zeros(1).to(self.device)).sum()  +  self.lambdaa_solution_regression* l2_reg
            
            self.optimizer_xsi.zero_grad()
       
            loss.backward()
            self.optimizer_xsi.step()
            

            with torch.no_grad():
                L_tot += loss

        if save_loss :
            pred_train = torch.argmin(self.pairwisedist(self.Network(X.to(self.device)), self.yref), dim=1).tolist()
            return L_tot, pred_train

    def reshape_liste_y(self, liste_y):
        lenght_max = max([len(element) for element in liste_y])
        y_train_shape_max = [element if len(element) == lenght_max else element + [element[-1]]*(lenght_max - len(element))                for element in liste_y]

        return y_train_shape_max


    def fit(self,  X_train, y_train, y_train_true=[] , X_val=[], y_val=[], val=False,  crit_val=2,      batch_size=200, **dict_params):
        
        self.val, self.crit_val = val, crit_val
        self.record_regression, self.record_xsi = [], []
        self.score_train, self.score_val_regression, self.score_val_xsi = [], [], []


        ######  Correction of size if partial label is not the same lenght everywhere
        
        y_train_shape_max = self.reshape_liste_y(liste_y=y_train)


        ######
        for self.epoch_regression in range(self.max_epoch_regression):

            self.optimize_regression(X=X_train, liste_y=y_train_shape_max, C=self.C, batch_size=batch_size, save_loss=False)
            #print(self.epoch_regression)

            if self.val :

                if self.epoch_regression % 10 == 0 :
                    loss, pred_train = self.optimize_regression(X=X_train, liste_y=y_train_shape_max, C=self.C, batch_size=200, save_loss=True)
                    self.record_regression.append([loss.detach().tolist(), self.score(y_test=y_train_true, y_pred=pred_train )])

                    if self.val :
                        self.score_val_regression.append(self.score(y_test=y_val, X_test=X_val)  )

                        if self.epoch_regression > 11 :
                            if self.score_val_regression[-2] - self.score_val_regression[-1] > self.crit_val :
                                print('early stopping regression')

                                break

        self.param_freeze = []
        for p in self.Network.parameters():
            self.param_freeze.append(p.detach())


        self.solution_regression = self.Network(X_train.to(self.device)).detach()

        for self.epoch_xsi in range(self.max_epoch_xsi):

            self.optimize_xsi(X=X_train, liste_y=y_train_shape_max, C=self.C, batch_size=batch_size, save_loss=False)

            if self.val : 
                if self.epoch_xsi % 10 ==0 :
                    loss, pred_train = self.optimize_xsi(X=X_train, liste_y=y_train_shape_max, C=self.C, batch_size=200, save_loss=True)
                    self.record_xsi.append([loss.detach().tolist(), self.score(y_test=y_train_true, y_pred=pred_train )])

                    if self.val :
                        self.score_val_xsi.append(self.score(y_test=y_val, X_test=X_val)  )

                        if self.epoch_xsi > 11 :
                            if self.score_val_xsi[-2] - self.score_val_xsi[-1] > self.crit_val :
                                print('early stopping optim xsi')
                                break
                

        return self


    def predict(self, X, y_prior=None):
        pairwise_dist = self.pairwisedist(self.Network(X.to(self.device)),self.yref)

        if y_prior is None:
            y_pred =  torch.argmin(pairwise_dist, dim=1).tolist()
        if y_prior is not None :
            y_pred = (torch.tensor(y_prior).to(self.device)).gather(1, pairwise_dist.gather(1, torch.tensor(y_prior).to(self.device)).topk(k=1, largest=False)[1]).reshape(-1).tolist()

        return y_pred

    def score(self, y_test, y_pred=None, X_test=None, y_prior=None):
        if y_pred is None:
            y_pred = self.predict(X_test, y_prior=y_prior)
        return f1_score(y_test, y_pred, average='micro')

    def error(self, y_test, y_pred=None, X_test=None, y_prior=None, C=torch.ones((253, 253))):
        if y_pred is None:
            y_pred = self.predict(X_test, y_prior=y_prior)

        vec_C_error = C[torch.tensor(y_pred).reshape(-1,1), torch.tensor(y_test).reshape(-1,1)]
        return (vec_C_error.sum()/torch.count_nonzero(vec_C_error)).detach().tolist()

    def param_grid(self):
        param_grid = {"lambdaa": [1.e-4, 1.e-3, 1.e-2, ],  # 3
                    "lambdaa_1": [ 1.e-3, 1.e-2, 5.e-2  ],  # 3
                        "lambdaa_solution_regression": [ 1.e-3, 1.e-2, 5.e-2 ], #4
                        "distance" : [
                                           # 'euclidian',
                                            'correlation'
                                            ]
                      }
        return param_grid

    def tiny_param_grid(self):
        param_grid = {"lambdaa": [1.e-4, ],  # 3
                    "lambdaa_1": [1.e-4, ],  # 7
                        "lambdaa_solution_regression": [1.e-4, 1.e-3 ], 
                        "distance" : [
                                            'euclidian',
                                            'correlation'
                                            ]
                              }
        return param_grid

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


    def get_params(self, deep=True):
            return { 
                 "lambdaa": self.lambdaa_,  # lambda regression ridge
                "lambdaa_solution_regression": self.lambdaa_solution_regression,  # pour P la loss mu*Lxsi - (1-mu)*L_reg
                "distance": self.distance
            }




class plSVM():

    def __init__(self, W=nn.Sequential(nn.Linear(26, 26)),
                 lambdaa=1e-5,  # regularisation

                 optimizer='Adam',
                 lr_P=1.e-5,
                 kernel=None,

                 epochs=200,

                 device='cpu',
                 C=torch.ones(26),
                 C_score=torch.ones(26)):

        self.device = device

        self.lambdaa = lambdaa
        self.max_epoch = epochs
        
        self.C = C
        self.C_score = C_score
        self.c = C.shape[0]
        self.lr_P = lr_P
        self.W = W.to(self.device)
        self.p = self.W[0].weight.shape[0]
        self.kernel = False
        if kernel:
            self.kernel = RBFSampler(gamma=1 / self.p, random_state=1, n_components=5 * self.p)
            self.W = nn.Sequential(nn.Linear(5 * self.p, self.c)).to(self.device)



        # Sequentiel
        if optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(params=self.W.parameters(), weight_decay=self.lambdaa)

        if optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(params=self.W.parameters(), weight_decay=self.lambdaa, lr=self.lr_P)

        self.C = self.C.to(self.device)

    def optimize_xsi(self, X, liste_y, set_Yiyilde, C, batch_size=200, save_loss=False):

        permutation = torch.randperm(X.size()[0])
        L_tot = 0
        for i in range(0, X.size()[0], batch_size):
            indices = permutation[i:i + batch_size]
            batch_Z = (X[indices]).to(self.device)
            batch_y = [liste_y[indice_y] for indice_y in indices]

            prod = self.W(batch_Z)

            prod_yi_max = torch.max(prod.gather(1, torch.tensor(batch_y)), dim=1)
            prod_Yi = prod_yi_max.values.reshape(-1, 1)
            real_indice_yi = torch.tensor(batch_y).gather(1, prod_yi_max.indices.reshape(-1, 1))


            prod_yi_comp_max = torch.max(prod.gather(1, set_Yiyilde[indices]), dim=1)
            prod_Yitilde = prod_yi_comp_max.values.reshape(-1, 1)
            real_indice_Yi_compl = set_Yiyilde[indices].gather(1, prod_yi_comp_max.indices.reshape(1, -1)).reshape(-1,1)


            vec_C_columns = self.C[real_indice_yi,:][:,0,:]
            new_vec_C_columns = vec_C_columns.gather(1, real_indice_Yi_compl )

            #all_xsi = torch.ones(batch_Z.size(0), 1) - (prod_Yi - prod_Yitilde)
            all_xsi = new_vec_C_columns - (prod_Yi - prod_Yitilde)
            loss = torch.max(all_xsi, torch.zeros(1)).sum()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                L_tot += loss

            if save_loss:
                pred = torch.argmax(self.W(X), dim=1).tolist()
                return L_tot, pred

    def reshape_liste_y(self, liste_y):
        lenght_max = max([len(element) for element in liste_y])
        y_train_shape_max = [
            element if len(element) == lenght_max else element + [element[-1]] * (lenght_max - len(element)) for element
            in liste_y]

        return y_train_shape_max

    def fit(self, X_train, y_train, y_train_true=[], X_val=[], y_val=[], val=False, crit_val=2, batch_size=200,
            **dict_params):

        if self.kernel :
            X_train = torch.FloatTensor(self.kernel.fit_transform(X_train))

        self.val, self.crit_val = val, crit_val
        self.record_regression, self.record_xsi = [], []
        self.score_train, self.score_val_regression, self.score_val_xsi = [], [], []

        ######  Correction of size if partial label is not the same lenght everywhere

        y_train_shape_max = self.reshape_liste_y(liste_y=y_train)
        reshape_tilde =self.reshape_liste_y(liste_y=[list(set(range(self.c)) - set(yi)) for yi in y_train_shape_max])
        set_Yiyilde = torch.tensor(reshape_tilde)

        ######
        for self.epoch in range(self.max_epoch):

            #set_Yiyilde =  torch.tensor([list(set(range(self.c)) - set(yi)) for yi in y_train_shape_max])

            self.optimize_xsi(X=X_train, liste_y=y_train_shape_max, set_Yiyilde=set_Yiyilde, C=self.C,
                              batch_size=batch_size, save_loss=False)

            if self.val :

                if self.epoch % 10 == 0:
                    loss, pred_train = self.optimize_xsi(X=X_train, liste_y=y_train_shape_max,set_Yiyilde=set_Yiyilde, C=self.C, batch_size=200,
                                                         save_loss=True)
                    self.record_regression.append(
                        [loss.detach().tolist(), self.score(y_test=y_train_true, y_pred=pred_train)])

                    if self.val:
                        self.score_val_regression.append(self.score(y_test=y_val, X_test=X_val))

                        if self.epoch > 11:
                            if self.score_val_regression[-2] - self.score_val_regression[-1] > self.crit_val:
                                print('early stopping regression')

                                break

        return self

    def predict(self, X, y_prior=None):

        if self.kernel :
            X = torch.FloatTensor(self.kernel.transform(X))

        if y_prior is None:
            y_pred = torch.argmax(self.W(X), dim=1).tolist()
        if y_prior is not None:
            y_pred = (torch.tensor(y_prior).gather(1, torch.argmax(self.W(X).gather(1, torch.tensor(y_prior)), dim=1).reshape(-1,1))).tolist()
        return y_pred

    def score(self, y_test, y_pred=None, X_test=None, y_prior=None):
        if y_pred is None:
            y_pred = self.predict(X_test, y_prior=y_prior)
        return f1_score(y_test, y_pred, average='micro')

    def error(self, y_test, y_pred=None, X_test=None, y_prior=None, C=torch.ones((253, 253))):
        if y_pred is None:
            y_pred = self.predict(X_test, y_prior=y_prior)

        vec_C_error = C[torch.tensor(y_pred).reshape(-1,1), torch.tensor(y_test).reshape(-1,1)]
        return (vec_C_error.sum()/torch.count_nonzero(vec_C_error)).detach().tolist()

    def param_grid(self):
        param_grid = {"lambdaa": [1.e-4, 1.e-3, 1.e-2, 1.e-3],  # 3
                      # "lambdaa_1": [1.e-4, 1.e-3, 1.e-2, 5.e-2  ],  # 7
                      #     "lambdaa_solution_regression": [1.e-4, 1.e-3, 1.e-2, 5.e-2 ],
                      #     "distance" : [
                      # 'euclidian',
                      #                        'correlation'
                      #                       ]
                      }
        return param_grid

    def tiny_param_grid(self):
        param_grid = {"lambdaa": [1.e-4, 1.e-3],  # 3

                      }
        return param_grid

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep=True):
        return {
            "lambdaa": self.lambdaa_,  # lambda regression ridge
            "lambdaa_solution_regression": self.lambdaa_solution_regression,  # pour P la loss mu*Lxsi - (1-mu)*L_reg
            "distance": self.distance
        }
