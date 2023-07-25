import torch
import torch.nn as nn
from sklearn.metrics import f1_score
import random
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
from pl_model import pl_nn_prototybe_based, pl_hKNN, plSVM
import os 

# python3 -u create_partial_label_prosstt.py
# %%


PATH = os.getcwd()
os.makedirs(PATH+'/data/datasets/final_Tree/', exist_ok=True)

modules = 50
value = 0.1
#topology = 'half'  ### CESETBON
lenght_tree = 21
for topology, lenght_tree in zip(['binary','linear', 'half'],[7,254,21]):
    overlap = 0

    # I = (1, 15)
    # I = (15, 253)
    for I in [(1, 15), (15, 253)] : 
        k = 4
        print('OVERLAP :',overlap)
        sub_proportion = 1 - 0.02
        save = True


        name_tree = '_'+str(modules)+'_'+str(value)




        path_file = PATH+'/data/datasets/final_Tree/' + str(topology) + '/' + str(lenght_tree) + '/'
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

     


        label_s, label_ws = train_test_split(range(c), test_size=0.5)
        ## Matrice qui nous donne quel label peut être associé aux autres par défault on sélectionne 12 labels par

        pairwise_label_probability = np.zeros((c,c))
        potentiel_labels = []
        for j in range(c):
            potentiel_labels.append(random.sample(list(set(torch.where(C[j] < I[1])[0].tolist())    & set(torch.where(C[j] >= I[0])[0].tolist())), k=12) )


        pairwise_label_probability = np.apply_along_axis(np.random.permutation, axis=1, arr=np.random.permutation(np.concatenate((np.ones((c,20)), np.zeros((c,c-20))), axis=1)))
        pairwise_label_probability = pairwise_label_probability + pairwise_label_probability.T
        for i in range(c):
            pairwise_label_probability[i,i]=0

        X_s, X_ws, y_s, y_ws, y_s_prior, y_ws_prior = [], [], [], [], [], []
        for i,yi in enumerate(y):
            if yi in label_s:
                y_s.append(yi)
                X_s.append(X[i])
                y_s_prior.append(
                    [yi] + random.sample(potentiel_labels[yi], k =9))
                    # random.sample(list(set(torch.where(C[yi] < I[1])[0].tolist())
                    #                    & set(torch.where(C[yi] >= I[0])[0].tolist())
                    #                    #& set(label_s)),
                    #                                 & set(np.where(pairwise_label_probability[yi] ==1)[0])),
                    #               k=9) )
            else:
                y_ws.append(yi)
                X_ws.append(X[i])
                y_ws_prior.append(
                    [yi] + random.sample(potentiel_labels[yi], k =9))
                    # random.sample(list(set(torch.where(C[yi] < I[1])[0].tolist())
                    #                    & set(torch.where(C[yi] >= I[0])[0].tolist())
                    #                    #& set(label_ws)),
                    #                    & set(np.where(pairwise_label_probability[yi] == 1)[0])),
                    #               k=9))


        print('Intersection label : ',len(set(y_s) & set(y_ws)), len(set(y)))

        name_to_save = PATH+'/data/datasets/final_Tree/' + str(topology) + '/' + str(lenght_tree)+'/' +str(overlap)+'_'+str(modules)+'_'+str(value)+'_'+str(I)+'_'         #'un_truc_qui_dépend_topology/modules_alpha/overlap_I/'
        if save :
            np.save(name_to_save+'X_s', X_s)
            np.save(name_to_save+'y_s', y_s)
            np.save(name_to_save+'y_s_prior', y_s_prior)
            np.save(name_to_save+ 'X_ws', X_ws),
            np.save(name_to_save+ 'y_ws', y_ws)
            np.save(name_to_save+ 'y_ws_prior', y_ws_prior)
            np.save(name_to_save+'potentiel_labels', potentiel_labels)

        #%% #%%  sauver les découpages train/test

        for t in range(5):
            X_train_s_0, X_test_s, y_train_0_s, y_test_s, indice_train_s, indice_test_s, y_train_0_s_prior, y_test_s_prior = train_test_split(torch.FloatTensor(X_s), y_s,
                                                                                               range(len(X_s)), y_s_prior,
                                                                                               test_size=0.165, stratify=y_s)
            X_train_ws_0, X_test_ws, y_train_0_ws, y_test_ws, indice_train_ws, indice_test_ws, y_train_0_ws_prior, y_test_ws_prior = train_test_split(torch.FloatTensor(X_ws), y_ws,
                                                                                               range(len(X_ws)), y_ws_prior,
                                                                                               test_size=0.165, stratify=y_ws)
            if save :
                np.save(name_to_save+ 'indice_train_s_'+str(t), indice_train_s)
                np.save(name_to_save + 'indice_test_s_' + str(t), indice_test_s)
                np.save(name_to_save + 'indice_train_ws_' + str(t), indice_train_ws)
                np.save(name_to_save + 'indice_test_ws_' + str(t), indice_test_ws)


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


            ### ON FAIT VARIER LE NOMBRE DE SUPERVISION
            sub_indice_train_s, _ = train_test_split(indice_train_s, test_size=sub_proportion, stratify=y_train_0_s)
            X_train_s, y_train_s, y_train_s_prior = torch.FloatTensor(X_s)[sub_indice_train_s], np.array(y_s)[sub_indice_train_s].tolist(), np.array(y_s_prior)[sub_indice_train_s].tolist()
            print('X_train_s_shape : ', X_train_s.shape)

            ## FIXE A 200 exemples par labels
            sub_indice_train_ws, _ = train_test_split(indice_train_ws, test_size=1-0.2, stratify=y_train_0_ws)
            X_train_ws, y_train_ws, y_train_ws_prior = torch.FloatTensor(X_ws)[sub_indice_train_ws], np.array(y_ws)[sub_indice_train_ws].tolist(), np.array(y_ws_prior)[sub_indice_train_ws].tolist()


            X_test_s, X_test_ws = torch.split(
                torch.FloatTensor(
                    preprocessing.StandardScaler().fit_transform(torch.cat((X_test_s, X_test_ws), dim=0))),
                [X_test_s.size(0), X_test_ws.size(0)])


            X_train_s, X_train_ws = torch.split(torch.FloatTensor(preprocessing.StandardScaler().fit_transform(torch.cat((X_train_s, X_train_ws), dim=0))), [X_train_s.size(0), X_train_ws.size(0)])
            y_train_s_prior, y_train_ws_prior = [element[:k] for element in y_train_s_prior], [element[:k] for element in y_train_ws_prior]

            print('y_train_ws_prior shape : ',np.shape(np.array(y_train_ws_prior)))
            


    overlap = 1 

    print('OVERLAP 1 ')

    for I in [(1, 15), (15, 253)] : 
        k = 4

        sub_proportion = 1 - 0.02
        save = True

        path_file_abs = PATH
        name_tree = '_'+str(modules)+'_'+str(value)

        path_file = PATH+'/data/datasets/final_Tree/' + str(topology) + '/' + str(lenght_tree) + '/'
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




        ## Matrice qui nous donne quel label peut être associé aux autres par défault on sélectionne 20 labels par

        pairwise_label_probability = np.zeros((c,c))
        potentiel_labels = []
        for j in range(c):
            potentiel_labels.append(random.sample(list(set(torch.where(C[j] < I[1])[0].tolist())    & set(torch.where(C[j] >= I[0])[0].tolist())), k=12) )

        y_prior = []

        for yi in y :
            y_prior.append([yi] + random.sample(potentiel_labels[yi], k =9 ))



        X_s, X_ws, y_s, y_ws, y_s_prior, y_ws_prior = train_test_split(X, y, y_prior, stratify=y, test_size = 0.5)
        #ça fait 605 échantillons dans chacun 




        print('Intersection label : ',len(set(y_s) & set(y_ws)), len(set(y)))
        


        name_to_save = PATH+'/data/datasets/final_Tree/' + str(topology) + '/' + str(lenght_tree)+'/' +str(overlap)+'_'+str(modules)+'_'+str(value)+'_'+str(I)+'_'         #'un_truc_qui_dépend_topology/modules_alpha/overlap_I/'
        if save :
            np.save(name_to_save+'X_s', X_s)
            np.save(name_to_save+'y_s', y_s)
            np.save(name_to_save+'y_s_prior', y_s_prior)
            np.save(name_to_save+ 'X_ws', X_ws),
            np.save(name_to_save+ 'y_ws', y_ws)
            np.save(name_to_save+ 'y_ws_prior', y_ws_prior)
            np.save(name_to_save+'potentiel_labels', potentiel_labels)

      #  sauver les découpages train/test

        for t in range(5):
            X_train_s_0, X_test_s, y_train_0_s, y_test_s, indice_train_s, indice_test_s, y_train_0_s_prior, y_test_s_prior = train_test_split(torch.FloatTensor(X_s), y_s,
                                                                                               range(len(X_s)), y_s_prior,
                                                                                               test_size=0.33, stratify=y_s)
            X_train_ws_0, X_test_ws, y_train_0_ws, y_test_ws, indice_train_ws, indice_test_ws, y_train_0_ws_prior, y_test_ws_prior = train_test_split(torch.FloatTensor(X_ws), y_ws,
                                                                                               range(len(X_ws)), y_ws_prior,
                                                                                               test_size=0.33, stratify=y_ws)

            #ça fait 199 échantillons dans le test et 406 dans le train
            if save :
                np.save(name_to_save+ 'indice_train_s_'+str(t), indice_train_s)
                np.save(name_to_save + 'indice_test_s_' + str(t), indice_test_s)
                np.save(name_to_save + 'indice_train_ws_' + str(t), indice_train_ws)
                np.save(name_to_save + 'indice_test_ws_' + str(t), indice_test_ws)


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


            ### ON FAIT VARIER LE NOMBRE DE SUPERVISION
            test_size_s=0.95
            if sub_proportion == 0.8 :
                test_size_s = 0.5
            sub_indice_train_s, _ = train_test_split(indice_train_s, test_size=test_size_s, stratify=y_train_0_s)
            X_train_s, y_train_s, y_train_s_prior = torch.FloatTensor(X_s)[sub_indice_train_s], np.array(y_s)[sub_indice_train_s].tolist(), np.array(y_s_prior)[sub_indice_train_s].tolist()
            print('X_train_s_shape : ', X_train_s.shape)

            ## FIXE A 200 exemples par labels
            sub_indice_train_ws, _ = train_test_split(indice_train_ws, test_size=0.5, stratify=y_train_0_ws)  ## 200 dans le train
            X_train_ws, y_train_ws, y_train_ws_prior = torch.FloatTensor(X_ws)[sub_indice_train_ws], np.array(y_ws)[sub_indice_train_ws].tolist(), np.array(y_ws_prior)[sub_indice_train_ws].tolist()


            X_test_s, X_test_ws = torch.split(
                torch.FloatTensor(
                    preprocessing.StandardScaler().fit_transform(torch.cat((X_test_s, X_test_ws), dim=0))),
                [X_test_s.size(0), X_test_ws.size(0)])


            X_train_s, X_train_ws = torch.split(torch.FloatTensor(preprocessing.StandardScaler().fit_transform(torch.cat((X_train_s, X_train_ws), dim=0))), [X_train_s.size(0), X_train_ws.size(0)])
            y_train_s_prior, y_train_ws_prior = [element[:k] for element in y_train_s_prior], [element[:k] for element in y_train_ws_prior]

            print('y_train_ws_prior shape : ',np.shape(np.array(y_train_ws_prior)))
            print('proportion vec ', y_train_s.count(0), y_train_ws.count(0))




