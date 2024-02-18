import os
import pandas as pd
import torch
from torchmetrics.functional import pairwise_cosine_similarity
import random
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from sklearn.model_selection import train_test_split


from sklearn import preprocessing
import torch
import torch.nn as nn
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

#%%

from sklearn.manifold import MDS
# import umap
from sklearn.manifold import Isomap
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import f1_score

from sklearn.kernel_approximation import RBFSampler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression




import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

def warn(*args, **kwargs):
    pass


import warnings
warnings.filterwarnings("ignore")





def random_between_argmax(a):
    return random.choice(np.flatnonzero(a == np.max(a)).tolist())



######################## Partial Label MODELS #####################"""""""

#### kNN

class pl_KNN:
    def __init__(self, k, C=[], C_score=[]):
        self.k = k
        self.C = C

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
        X_test = X

        pairwisedist = torch.cdist(self.X_train, X_test).to('cpu')
        topk = torch.topk(pairwisedist.T, largest=False, k=self.k).indices  # indices des kplus proches voisins

        ytopk = torch.tensor(self.y_train)[topk]
        ytopk = ytopk.reshape(ytopk.shape[0], -1)
        result = torch.mode(ytopk, dim=1)[0].detach().tolist()
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
        param_grid = {"k": [5,10, 20, 50, 100],  # 5
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


###### Hierarchical kNN

class pl_hKNN():
    def __init__(self, k=10, C=[], C_score=[], flat=True):
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

            if self.flat:

                if y_prior is None :
                    result = result + torch.mode(ytopk, dim=1)[0].detach().tolist()
                else:
                    #print(y_prior)
                    batch_prior = torch.tensor(y_prior)[i:i + batch_size].detach().tolist()
                    
                    count_element_prior = [[Counter(liste_i.detach().tolist())[candidat] for candidat in batch_prior[i]] for
                                           i, liste_i in enumerate(ytopk)]
                    result_prior = [batch_prior[index][random_between_argmax(count_element)] for index, count_element in
                                    enumerate(count_element_prior)]

                    result = result + result_prior

            else:

                all_votes = (1 / 2 ** (self.C[:, ytopk])).sum(dim=2)

                if y_prior is None:
                    argmax_votes = torch.argmax(all_votes, dim=0)
                    result = result + argmax_votes.tolist()

                if y_prior is not None:
                    batch_prior = torch.tensor(y_prior)[i:i + batch_size]
                    all_votes_prior = torch.tensor([(all_votes[i, ix]).tolist() for ix, i in enumerate(batch_prior)])
                    argmax_prior_votes = torch.argmax(all_votes_prior, dim=1)
                    liste_result = [(batch_prior[i][argmax_prior_votes[i]]).tolist() for i in
                                    range(argmax_prior_votes.shape[0])]
                    result = result + liste_result

        return result

    def score(self, y_test, X_test=None, y_prior=None, y_pred=None):
        if y_pred is None:
            y_pred = self.predict(X_test)
        return f1_score(y_pred, y_test, average='micro')

    def error(self, y_test, y_pred=None, X_test=None, y_prior=None, C=torch.ones((253, 253))):
        if y_pred is None:
            y_pred = self.predict(X_test, y_prior=y_prior)

        vec_C_error = C[torch.tensor(y_pred).reshape(-1, 1), torch.tensor(y_test).reshape(-1, 1)]
        return (vec_C_error.sum() / torch.count_nonzero(vec_C_error)).detach().tolist()

    def param_grid(self):
        param_grid = {"k": [1, 10, 20, 50, 100],  # 5
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

                     epochs_regression = 500,
                     epochs_xsi=2000, 
                      
                     distance = 'euclidian',
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
                 gamma = 1/1000,
                 p=500,

                 epochs=200,

                 device='cpu',
                 C=torch.ones(26),
                 C_score=torch.ones(26)):

        self.device = device

        self.lambdaa = lambdaa
        self.max_epoch = epochs
        self.gamma = gamma
        self.dim_out = p
        
        self.C = C
        self.C_score = C_score
        self.c = C.shape[0]
        self.lr_P = lr_P
        self.W = W.to(self.device)
        self.p = self.W[0].weight.shape[0]
        
        self.kernel = False
        if kernel:
            self.kernel = RBFSampler(gamma=self.gamma, random_state=1, n_components= self.dim_out)
            self.W = nn.Sequential(nn.Linear(self.dim_out, self.c)).to(self.device)



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
        param_grid = {"lambdaa": np.logspace(-4,1,5).tolist(), #[1.e-4, 1.e-3, 1.e-2],  # 3
                          "gamma": [1],  # 20
                          "p": [1] # 3
                   
                      }
        if self.kernel :
           param_grid = {"lambdaa": np.logspace(-4,1,5).tolist(),  # 2
                      "gamma": np.logspace(-4,1,10).tolist(),  # 20
                          "p": [500, 1000],}
        return param_grid

    def tiny_param_grid(self):
        param_grid = {"lambdaa": [1.e-4, 1.e-3], 
         "gamma": [1],  # 20
                          "p": [1] # 3

                      }
        if self.kernel :
            param_grid = {"lambdaa": [1.e-4, 1.e-3],  # 2
                      "gamma": np.logspace(-4,1,20).tolist(),  # 20
                          "p": [500],
                    
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


### Logistic REgression 

class plLR():

    def __init__(self, W=nn.Sequential(nn.Linear(26, 26), nn.Softmax(dim=1)),
                 lambdaa=1e-5,  # regularisation

                 optimizer='Adam',
                 lr_P=1.e-4,
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
        self.sigmoid = nn.Sigmoid()

        self.loss = nn.CrossEntropyLoss()


        self.p = self.W[0].weight.shape[0]
        self.kernel = False
        if kernel:
            self.kernel = RBFSampler(gamma=1 / self.p, random_state=1, n_components=200 * self.p)
            self.W = nn.Sequential(nn.Linear(200 * self.p, self.c)).to(self.device)



        # Sequentiel
        if optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(params=self.W.parameters(), weight_decay=self.lambdaa)

        if optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(params=self.W.parameters(), weight_decay=self.lambdaa, lr=self.lr_P)

        self.C = self.C.to(self.device)

    def optimize(self, X, liste_y, batch_size=200, save_loss=False):

        permutation = torch.randperm(X.size()[0])
        L_tot = 0
        Y = torch.tensor(liste_y)
        for i in range(0, X.size()[0], batch_size):
            indices = permutation[i:i + batch_size]
            batch_Z = (X[indices]).to(self.device)
            batch_Y = (Y[indices]).to(self.device)
            
            
            self.optimizer.zero_grad()

            self.l = self.loss(self.W(batch_Z), batch_Y)
            self.l.backward()

            self.optimizer.step()

    
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


        y_train_shape_max = self.reshape_liste_y(liste_y=y_train)
        reshape_tilde =self.reshape_liste_y(liste_y=[list(set(range(self.c)) - set(yi)) for yi in y_train_shape_max])
        set_Yiyilde = torch.tensor(reshape_tilde)


        z_train = self.predict(X=X_train, y_prior=y_train_shape_max)


        for self.epoch in range(self.max_epoch):


            #print(z_train)
            self.optimize(X=X_train, liste_y=z_train, batch_size=batch_size, save_loss=False)
            z_train = self.predict(X=X_train, y_prior=y_train_shape_max)


        return self

    def predict(self, X, y_prior=None):

        if self.kernel :
            X = torch.FloatTensor(self.kernel.transform(X))

        if y_prior is None:
            y_pred = torch.argmax(self.W(X), dim=1).tolist()
        if y_prior is not None:
            yprior =  torch.tensor(y_prior) #.reshape(-1,1)
            pred = yprior.gather(dim=1, index= self.W(X).gather(1, yprior).max(dim=1).indices.reshape(-1,1)  )
            #print(pred)
            y_pred =  [yi.detach().tolist()[0] for yi in pred]
            #y_pred = (torch.tensor(y_prior).gather(1, torch.argmax(self.W(X).gather(1, torch.tensor(y_prior)), dim=1).reshape(-1,1))).tolist()
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
        param_grid = {"lambdaa": [1.e-6, 5.e-6, 1.e-5, 5.e-5, 1.e-4, 5.e-4, 1.e-3, 5e-3, 1.e-2, 5.e-2],  # 3

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


### Radom Forest (algo IFR)

class plRF():

    def __init__(self,
                 n_estimators=50,
                 max_depht=20,
                 criterion='gini',
                 max_features='sqrt',
                 epochs=50,
                 indice_network=0,
                 device='cpu',
                 C=torch.ones(26),
                 C_score=torch.ones(26),
                 supervised=False
                 ):

        self.device = device

        self.supervised = supervised
        self.n_estim = n_estimators
        self.max_depht = max_depht
        self.max_epoch = epochs

        self.criterion = criterion
        self.max_features = max_features

        self.model = RandomForestClassifier(
            n_estimators=self.n_estim,
            max_depth=self.max_depht,
            criterion=self.criterion,
            max_features=self.max_features
        )

        self.C = C
        self.C_score = C_score
        self.c = C.shape[0]
        self.model.classes_ = self.c
        self.model.n_classes_ = self.c
        # print(self.model.classes_)
        self.C = self.C.to(self.device)
        self.indice_network = indice_network

        self.LE = LabelEncoder()

    def reshape_liste_y(self, liste_y):
        lenght_max = max([len(element) for element in liste_y])
        y_train_shape_max = [
            element if len(element) == lenght_max else element + [element[-1]] * (lenght_max - len(element)) for element
            in liste_y]

        return y_train_shape_max

    def fit(self, X_train, y_train, y_train_true=[], X_val=[], y_val=[], val=False, crit_val=2, batch_size=200,
            **dict_params):

        if self.supervised == True:
            self.LE.fit(y_train)
            self.model.fit(X=X_train, y=[element[0] for element in y_train])



        else:

            self.LE.fit(list(itertools.chain.from_iterable(y_train)))

            y_train_shape_max = self.reshape_liste_y(liste_y=y_train)
            # print(y_train_shape_max)
            z_train_random = [random.choice(self.LE.transform(i)) for i in y_train_shape_max]

            self.LE_t = LabelEncoder()
            z_train_random = self.LE_t.fit_transform(z_train_random).tolist()

            self.model.fit(X=X_train, y=z_train_random)
           
            ######
            for self.epoch in range(self.max_epoch):
                # Prediction is on c label
                z_train = self.predict(X=X_train, y_prior=y_train_shape_max)
                z_encoded = self.LE.transform(z_train)
                self.LE_t = LabelEncoder()
                z_encoded = self.LE_t.fit_transform(z_encoded).tolist()
                self.model.fit(X=X_train, y=z_encoded)


        return self

    def predict(self, X, y_prior=None):

        if y_prior is None:
            y_pred = self.model.predict(X).tolist()  # torch.argmax(self.W(X), dim=1).tolist()
            y_pred = self.LE_t.inverse_transform(y_pred)

            y_pred = self.LE.inverse_transform(y_pred).tolist()

        if y_prior is not None:
            proba_predict = self.model.predict_proba(X)
            nb_classes_predict = proba_predict.shape[1]
            # print(nb_classes_predict)

            indice_y_prior = self.LE_t.inverse_transform(range(nb_classes_predict))
            # print("1er inverse transofrm",indice_y_prior )

            indice_y_prior = self.LE.inverse_transform(indice_y_prior).tolist()
            # print("2er inverse transofrm", indice_y_prior)

            matrice_proba = np.zeros((X.shape[0], self.c))
            matrice_proba[:, indice_y_prior] = proba_predict
            # print(matrice_proba)
            #y_pred = [i[matrice_proba[e][i].argmax()] for e, i in enumerate(y_prior)]
            y_pred = [i[random_between_argmax(matrice_proba[e, i])] for e, i in enumerate(y_prior)]
        return y_pred

    def score(self, y_test, y_pred=None, X_test=None, y_prior=None):
        if y_pred is None:
            y_pred = self.predict(X_test, y_prior=y_prior)
        return f1_score(y_test, y_pred, average='micro')

    def error(self, y_test, y_pred=None, X_test=None, y_prior=None, C=torch.ones((253, 253))):
        if y_pred is None:
            y_pred = self.predict(X_test, y_prior=y_prior)

        vec_C_error = C[torch.tensor(y_pred).reshape(-1, 1), torch.tensor(y_test).reshape(-1, 1)]
        return (vec_C_error.sum() / torch.count_nonzero(vec_C_error)).detach().tolist()

    def param_grid(self):
        param_grid = {'n_estimators': [50, 100,150, 200],
                      'max_features': [ 'sqrt'],
                      'max_depht': [5, 10, 15,20],
                      'criterion': ['entropy']
                      }
        return param_grid

    def tiny_param_grid(self):
        param_grid = {'n_estimators': [50],
                      'max_features': ['auto', ],
                      'max_depth': [5],
                      'criterion': ['gini']}
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



### kernel SVM with scikitlearn implementation + real kernel computation (algo IFR)




###Extreme Gradient Boosting method

class plXGBM():
    def __init__(self,
                 # param XGBOOST
                 n_estimators=50,
                 max_depht=10,
                 learning_rate=0.01,
                 gamma=1,
                 min_child_weight=1,

                 # param PL algo
                 epochs=10,
                 supervised=False,

                 # param for metric and computation
                 indice_network=0,
                 device='cpu',

                 C=torch.ones(26),
                 C_score=torch.ones(26),

                 ):

        self.device = device
        self.supervised = supervised

        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.n_estimators = n_estimators
        self.max_depht = max_depht
        self.max_epoch = epochs

        self.model = xgb.XGBClassifier(

            learning_rate=self.learning_rate,

            n_estimators=self.n_estimators,
            max_depth=self.max_depht,
            min_child_weight=self.min_child_weight,
            verbosity=0,
            silent=True,
            nthread=1,
            objective='binary:logistic',
        )

        self.C = C
        self.C_score = C_score
        self.c = C.shape[0]
        # self.model.classes_ = self.c
        # self.model.n_classes_ = self.c
        # print(self.model.classes_)
        self.C = self.C.to(self.device)
        self.indice_network = indice_network
        self.LE = LabelEncoder()
        self.count= []



    def reshape_liste_y(self, liste_y):
        lenght_max = max([len(element) for element in liste_y])
        y_train_shape_max = [
            element if len(element) == lenght_max else element + [element[-1]] * (lenght_max - len(element)) for element
            in liste_y]

        return y_train_shape_max

    def fit(self, X_train, y_train, y_train_true=[], X_val=[], y_val=[], val=False, crit_val=2, batch_size=200,
            **dict_params):


        if self.supervised == True:
            self.LE = LabelEncoder()
            y_train = self.LE.fit_transform(y_train).tolist()
            self.LE_t = LabelEncoder.fit_transform(y_train).tolist()
            self.model.fit(X=X_train, y=y_train)
            # [element[0] for element in y_train])


        else:

            ######  Correction of size if partial label is not the same lenght everywhere


            self.LE.fit(list(itertools.chain.from_iterable(y_train)))

            y_train_shape_max = self.reshape_liste_y(liste_y=y_train)
            #print(y_train_shape_max)
            z_train_random = [random.choice(self.LE.transform(i)) for i in y_train_shape_max]


            self.LE_t = LabelEncoder()
            z_train_random = self.LE_t.fit_transform(z_train_random).tolist()
            self.model.fit(X=X_train, y=z_train_random)
            #print(0, self.score(y_test=y_val, X_test=X_val))

            ######
            for self.epoch in range(self.max_epoch):

                # Prediction is on c label
                z_train = self.predict(X=X_train , y_prior=y_train_shape_max)
                z_encoded = self.LE.transform(z_train)
                self.LE_t = LabelEncoder()
                z_encoded = self.LE_t.fit_transform(z_encoded).tolist()
                self.model.fit(X=X_train, y=z_encoded)
                #print(self.epoch, self.score(y_test=y_val, X_test=X_val) )

        return self

    def predict(self, X, y_prior=None):


        if y_prior is None:
            y_pred = self.model.predict(X).tolist()  # torch.argmax(self.W(X), dim=1).tolist()
            y_pred = self.LE_t.inverse_transform(y_pred)

            y_pred = self.LE.inverse_transform(y_pred).tolist()

        if y_prior is not None:


            proba_predict = self.model.predict_proba(X)
            nb_classes_predict = proba_predict.shape[1]
            #print(nb_classes_predict)

            indice_y_prior = self.LE_t.inverse_transform(range(nb_classes_predict))
            #print("1er inverse transofrm",indice_y_prior )

            indice_y_prior = self.LE.inverse_transform(indice_y_prior).tolist()
            #print("2er inverse transofrm", indice_y_prior)

            matrice_proba = np.zeros((X.shape[0], self.c))
            matrice_proba[:, indice_y_prior] = proba_predict
            #print(matrice_proba)
            y_pred = [i[matrice_proba[e][i].argmax()] for e, i in enumerate(y_prior)]


        return y_pred

    def score(self, y_test, y_pred=None, X_test=None, y_prior=None):
        if y_pred is None:
            y_pred = self.predict(X_test, y_prior=y_prior)
            # y_pred = self.LE_t.inverse_transform(y_pred).tolist()
            # y_pred = self.LE.inverse_transform(y_pred).tolist()
        return f1_score(y_test, y_pred, average='micro')

    def error(self, y_test, y_pred=None, X_test=None, y_prior=None, C=torch.ones((253, 253))):
        if y_pred is None:
            y_pred = self.predict(X_test, y_prior=y_prior)
            y_pred = self.LE.inverse_transform(y_pred).tolist()
        vec_C_error = C[torch.tensor(y_pred).reshape(-1, 1), torch.tensor(y_test).reshape(-1, 1)]
        return (vec_C_error.sum() / torch.count_nonzero(vec_C_error)).detach().tolist()

    def param_grid(self):
        param_grid = {'n_estimators': [10, 20, 50],
                      'learning_rate': [0.1, 0.05, 0.01],
                      'max_depht': [10, 15, 30],
                      # 'criterion' :['gini', 'entropy']

                      'min_child_weight': [1],
                      'gamma': [0.5],
                      'subsample': [0.8],
                      'colsample_bytree': [1.0],

                      }
        return param_grid

    def tiny_param_grid(self):
        param_grid = {'n_estimators': [5],
                      'learning_rate': [0.1, 0.05, 0.01],
                      'max_depth': [3],
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






class plkernelSVC():

    def __init__(self,
                 lambdaa=10,
                gamma=1/50,
                 epochs=20,
                 indice_network=2,
                 device='cpu',
                 C=torch.ones(26),
                 C_score=torch.ones(26),
                 supervised=False
                 ):

        self.device = device

        self.supervised = supervised
        self.lambdaa = lambdaa
        self.gamma = gamma
        self.model = SVC(probability=True,
                         verbose=False,
                         kernel='rbf',
                         gamma=self.gamma,
                         C=self.lambdaa,
                         max_iter=-1)

        self.max_epoch = epochs
        self.C = C
        self.C_score = C_score
        self.c = C.shape[0]
        self.model.classes_ = self.c
        self.model.n_classes_ = self.c
        # print(self.model.classes_)
        self.C = self.C.to(self.device)
        self.indice_network = indice_network
        self.LE = LabelEncoder()

    def reshape_liste_y(self, liste_y):
        lenght_max = max([len(element) for element in liste_y])
        y_train_shape_max = [
            element if len(element) == lenght_max else element + [element[-1]] * (lenght_max - len(element)) for element
            in liste_y]

        return y_train_shape_max


    def fit(self, X_train, y_train, y_train_true=[], X_val=[], y_val=[], val=False, crit_val=2, batch_size=200,
            **dict_params):

        if self.supervised == True:
            self.LE.fit(y_train)
            self.model.fit(X=X_train, y=[element[0] for element in y_train])



        else:

            self.LE.fit(list(itertools.chain.from_iterable(y_train)))

            y_train_shape_max = self.reshape_liste_y(liste_y=y_train)
            # print(y_train_shape_max)
            z_train_random = [random.choice(self.LE.transform(i)) for i in y_train_shape_max]

            self.LE_t = LabelEncoder()
            z_train_random = self.LE_t.fit_transform(z_train_random).tolist()

            self.model.fit(X=X_train, y=z_train_random)

            ######
            for self.epoch in range(self.max_epoch):
                # Prediction is on c label
                z_train = self.predict(X=X_train, y_prior=y_train_shape_max)
                z_encoded = self.LE.transform(z_train)
                self.LE_t = LabelEncoder()
                z_encoded = self.LE_t.fit_transform(z_encoded).tolist()
                self.model.fit(X=X_train, y=z_encoded)

        return self

    def predict(self, X, y_prior=None):

        if y_prior is None:
            y_pred = self.model.predict(X).tolist()  # torch.argmax(self.W(X), dim=1).tolist()
            y_pred = self.LE_t.inverse_transform(y_pred)

            y_pred = self.LE.inverse_transform(y_pred).tolist()

        if y_prior is not None:
            proba_predict = self.model.predict_proba(X)
            nb_classes_predict = proba_predict.shape[1]

            indice_y_prior = self.LE_t.inverse_transform(range(nb_classes_predict))

            indice_y_prior = self.LE.inverse_transform(indice_y_prior).tolist()

            matrice_proba = np.zeros((X.shape[0], self.c))
            matrice_proba[:, indice_y_prior] = proba_predict

            y_pred = [i[random_between_argmax(matrice_proba[e, i])] for e, i in enumerate(y_prior)]
        return y_pred



    def score(self, y_test, y_pred=None, X_test=None, y_prior=None):
        if y_pred is None:
            y_pred = self.predict(X_test, y_prior=y_prior)
        return f1_score(y_test, y_pred, average='micro')

    def error(self, y_test, y_pred=None, X_test=None, y_prior=None, C=torch.ones((253, 253))):
        if y_pred is None:
            y_pred = self.predict(X_test, y_prior=y_prior)

        vec_C_error = C[torch.tensor(y_pred).reshape(-1, 1), torch.tensor(y_test).reshape(-1, 1)]
        return (vec_C_error.sum() / torch.count_nonzero(vec_C_error)).detach().tolist()

    def param_grid(self):
        param_grid = {'lambdaa': [1, 10, 100, 1000],
                      'gamma':np.logspace(-4,2,10).tolist() ,
                      }
        return param_grid

    def tiny_param_grid(self):
        param_grid = {'lambdaa': [1, 10, 100, 1000],
                        'gamma':np.logspace(-4,2,10).tolist()
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





### Logistic Regression (algo IFR)


class plEMLR():

    def __init__(self,

                 W=nn.Sequential(nn.Linear(26, 26),nn.Softmax(dim=1)),
                 lambdaa=1e-5,  # regularisation

                 lr_P=1.e-5,
                 kernel=None,

                 epochs=10,
                 supervised=False,

                 device='cpu',
                 C=torch.ones(26),
                 C_score=torch.ones(26)):

        self.lambdaa = lambdaa
        self.max_epoch = epochs
        self.W = W

        self.C = C
        self.C_score = C_score
        self.c = C.shape[0]
        self.device = device
        self.C = self.C.to(self.device)

        self.model = plLR(
            W=self.W,
            lambdaa=self.lambdaa,  # regularisation

            optimizer='Adam',
            lr_P=1.e-5,
            kernel=None,

            epochs=200,

            device='cpu',
            C=self.C,
            C_score=self.C_score)

        self.max_epoch = epochs

        self.supervised = supervised

        # print(self.model.classes_)

        self.indice_network = 0
        self.LE = LabelEncoder()

    def reshape_liste_y(self, liste_y):
        lenght_max = max([len(element) for element in liste_y])
        y_train_shape_max = [
            element if len(element) == lenght_max else element + [element[-1]] * (lenght_max - len(element)) for element
            in liste_y]

        return y_train_shape_max

    def fit(self, X_train, y_train, y_train_true=[], X_val=[], y_val=[], val=False, crit_val=2, batch_size=200,
            **dict_params):

        if self.supervised == True:
            self.LE.fit(y_train)
            self.model.fit(X_train=X_train, y_train=y_train)


        else:

            y_train_shape_max = self.reshape_liste_y(liste_y=y_train)

            # z_train_random = [[random.choice(self.LE.transform(i))] for i in y_train_shape_max]
            z_train_random = [[random.choice(i)] for i in y_train_shape_max]

            self.model.fit(X_train=X_train, y_train=z_train_random)

            ######
            for self.epoch in range(self.max_epoch):
                z_train = self.predict(X=X_train, y_prior=y_train_shape_max)
                self.model = plLR(
                    W=self.W,
                    lambdaa=self.lambdaa,  # regularisation
                    optimizer='Adam',
                    lr_P=1.e-5,
                    kernel=None,
                    epochs=200,
                    device='cpu',
                    C=self.C,
                    C_score=self.C_score)

                z_encoded = [[zi] for zi  in z_train ]

                self.model.fit(X_train=X_train, y_train=z_encoded)

        return self

    def predict(self, X, y_prior=None):

        if y_prior is None:
            y_pred = self.model.predict(X)  # torch.argmax(self.W(X), dim=1).tolist()
            # y_pred = self.LE.inverse_transform(y_pred).tolist()
        if y_prior is not None:
            y_pred = self.model.predict(X=X, y_prior=y_prior)

        return y_pred

    def score(self, y_test, y_pred=None, X_test=None, y_prior=None):
        if y_pred is None:
            y_pred = self.predict(X_test, y_prior=y_prior)
        return f1_score(y_test, y_pred, average='micro')

    def error(self, y_test, y_pred=None, X_test=None, y_prior=None, C=torch.ones((253, 253))):
        if y_pred is None:
            y_pred = self.predict(X_test, y_prior=y_prior)

        vec_C_error = C[torch.tensor(y_pred).reshape(-1, 1), torch.tensor(y_test).reshape(-1, 1)]
        return (vec_C_error.sum() / torch.count_nonzero(vec_C_error)).detach().tolist()

    def param_grid(self):
        param_grid = {"lambdaa": [1.e-6, 5.e-6, 1.e-5, 5.e-5, 1.e-4, 5.e-4, 1.e-3, 5e-3, 1.e-2, 5.e-2],  # 3

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




### SVM (algo IFR)




class plEMSVM():

    def __init__(self,

                W=nn.Sequential(nn.Linear(26, 26)),
                lambdaa=1e-5,  # regularisation

               
                 lr_P=1.e-5,
                 kernel=None,

                 epochs=200,
                 supervised = False,

                 device='cpu',
                 C=torch.ones(26),
                 C_score=torch.ones(26)):


        self.lambdaa = lambdaa
        self.max_epoch = epochs
        self.W = W

        self.C = C
        self.C_score = C_score
        self.c = C.shape[0]
        self.device = device
        self.C = self.C.to(self.device)
        

        self.model = plSVM(

                W=self.W,
                lambdaa=self.lambdaa,  # regularisation

                optimizer='Adam',
                lr_P=1.e-5,
                kernel=None,

                epochs=200,

                 device='cpu',
                 C=self.C,
                 C_score=self.C_score )

        self.max_epoch = epochs
        
        
        self.supervised = supervised

        
        #print(self.model.classes_)
        
        self.indice_network=0
        self.LE = LabelEncoder()



    def reshape_liste_y(self, liste_y):
        lenght_max = max([len(element) for element in liste_y])
        y_train_shape_max = [
            element if len(element) == lenght_max else element + [element[-1]] * (lenght_max - len(element)) for element
            in liste_y]

        return y_train_shape_max

    def fit(self, X_train, y_train, y_train_true=[], X_val=[], y_val=[], val=False, crit_val=2, batch_size=200,
            **dict_params):

        if self.supervised == True:
            self.LE.fit(y_train)
            self.model.fit(X_train=X_train, y_train=y_train)



        else:

            #self.LE.fit(list(itertools.chain.from_iterable(y_train)))
            y_train_shape_max = self.reshape_liste_y(liste_y=y_train)

            #z_train_random = [[random.choice(self.LE.transform(i))] for i in y_train_shape_max]
            z_train_random = [[random.choice(i)] for i in y_train_shape_max]

            self.model.fit(X_train=X_train, y_train=z_train_random)

            ######
            for self.epoch in range(self.max_epoch):
                z_train = self.predict(X=X_train, y_prior=y_train_shape_max)
                #z_encoded = self.LE.transform(z_train)
                z_encoded = z_train
                
                self.model.fit(X_train=X_train, y_train=z_encoded)


        return self

    def predict(self, X, y_prior=None):


        if y_prior is None:
            y_pred = self.model.predict(X) # torch.argmax(self.W(X), dim=1).tolist()
            #y_pred = self.LE.inverse_transform(y_pred).tolist()
        if y_prior is not None:


            y_pred = self.model.predict(X=X, y_prior=y_prior)

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
        
                      }
        return param_grid

    def tiny_param_grid(self):
        param_grid = {'lambdaa': [1.e-4, 1.e-3, 1.e-2, 1.e-3]}
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




