import os
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn import preprocessing
import torch
import random


save = True

PATH = os.getcwd()
rseed = 1312

for dataset in ['Planaria', 'Paul', 'Packer']:
    print(dataset)
    if dataset == 'Packer':
        path_file = PATH + '/data/datasets/' + str(dataset) + '/'
        try:
            X = np.load(path_file + 'X_pca.npy', allow_pickle=True)
            y = np.load(path_file + 'y.npy', allow_pickle=True).tolist()
            mat_dist = np.load(path_file + str(dataset) + '_mat_dist.npy')
            print('X,y loaded')
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
        try:
            path_file = PATH + '/data/datasets/' + str(dataset) + '/'
            X = np.load(path_file + 'sample/X.npy', allow_pickle=True)
            y = np.load(path_file + 'sample/y.npy', allow_pickle=True).tolist()

            mat_dist = np.load(path_file + str(dataset) + '_mat_dist.npy')
        except:
            path_file = os.getcwd() + '/data/datasets/' + str(dataset) + '/'
            X = np.load(path_file + 'sample/X.npy', allow_pickle=True)
            y = np.load(path_file + 'sample/y.npy', allow_pickle=True).tolist()

            mat_dist = np.load(path_file + str(dataset) + '_mat_dist.npy')

        if dataset == 'Paul':
            X, y = X[:-1], y[:-1]

    C = torch.Tensor(mat_dist)
    X = np.vstack(X).astype(np.float64)
    # X= torch.FloatTensor(X)

    c = C.shape[0]

    overlap = 0

    # I = (1, 15)
    # I = (15, 253)
    for I in ['I0', 'I1']:
        strI = I
        if I == 'I0':
            I = (1, 4)
            if dataset == 'Packer':
                I = (1, 8)
        if I == 'I1':
            I = (4, 9)
            if dataset == 'Packer':
                I = (8, 16)

        print(I)

        label_s, label_ws = train_test_split(range(c), test_size=0.5)
        ## Matrice qui nous donne quel label peut être associé aux autres par défault on sélectionne 20 labels par
        # %%
        pairwise_label_probability = np.zeros((c, c))
        potentiel_labels = []
        for j in range(c):

            # potentiel_labels.append(random.sample(list(set(torch.where(C[j] < I[1])[0].tolist())    & set(torch.where(C[j] >= I[0])[0].tolist())), k=12) )
            set_possible = list(set(torch.where(C[j] < I[1])[0].tolist()) & set(torch.where(C[j] >= I[0])[0].tolist()))
            if len(set_possible) > 12:
                potentiel_labels.append(random.sample(
                    list(set(torch.where(C[j] < I[1])[0].tolist()) & set(torch.where(C[j] >= I[0])[0].tolist())), k=12))

            else:
                potentiel_labels.append(set_possible + list(
                    random.sample(list(set(range(c)) - set(set_possible)), k=12 - len(set_possible))))
        # pairwise_label_probability = np.apply_along_axis(np.random.permutation, axis=1, arr=np.random.permutation(np.concatenate((np.ones((c,20)), np.zeros((c,c-20))), axis=1)))
        # pairwise_label_probability = pairwise_label_probability + pairwise_label_probability.T
        # for i in range(c):
        #     pairwise_label_probability[i,i]=0

        # %%

        # %%
        X_s, X_ws, y_s, y_ws, y_s_prior, y_ws_prior = [], [], [], [], [], []
        for i, yi in enumerate(y):
            if yi in label_s:
                y_s.append(yi)
                X_s.append(X[i])
                y_s_prior.append(
                    [yi] + random.sample(potentiel_labels[yi], k=9))
                # random.sample(list(set(torch.where(C[yi] < I[1])[0].tolist())
                #                    & set(torch.where(C[yi] >= I[0])[0].tolist())
                #                    #& set(label_s)),
                #                                 & set(np.where(pairwise_label_probability[yi] ==1)[0])),
                #               k=9) )
            else:
                y_ws.append(yi)
                X_ws.append(X[i])
                y_ws_prior.append(
                    [yi] + random.sample(potentiel_labels[yi], k=9))
                # random.sample(list(set(torch.where(C[yi] < I[1])[0].tolist())
                #                    & set(torch.where(C[yi] >= I[0])[0].tolist())
                #                    #& set(label_ws)),
                #                    & set(np.where(pairwise_label_probability[yi] == 1)[0])),
                #               k=9))



        print('Intersection label : ', len(set(y_s) & set(y_ws)), len(set(y)))

        path_file = PATH + '/data/datasets/' + str(dataset) + '/'

        name_to_save = path_file + str(overlap) + '_' + str(
            strI) + '_'  # 'un_truc_qui_dépend_topology/modules_alpha/overlap_I/'
        if save:
            np.save(name_to_save + 'X_s', X_s)
            np.save(name_to_save + 'y_s', y_s)
            np.save(name_to_save + 'y_s_prior', y_s_prior)
            np.save(name_to_save + 'X_ws', X_ws),
            np.save(name_to_save + 'y_ws', y_ws)
            np.save(name_to_save + 'y_ws_prior', y_ws_prior)
            np.save(name_to_save + 'potentiel_labels', potentiel_labels)



        for t in range(5):
            X_train_s_0, X_test_s, y_train_0_s, y_test_s, indice_train_s, indice_test_s, y_train_0_s_prior, y_test_s_prior = train_test_split(
                torch.FloatTensor(X_s), y_s,
                range(len(X_s)), y_s_prior,
                test_size=0.165, stratify=y_s)
            X_train_ws_0, X_test_ws, y_train_0_ws, y_test_ws, indice_train_ws, indice_test_ws, y_train_0_ws_prior, y_test_ws_prior = train_test_split(
                torch.FloatTensor(X_ws), y_ws,
                range(len(X_ws)), y_ws_prior,
                test_size=0.165, stratify=y_ws)
            if save:
                np.save(name_to_save + 'indice_train_s_' + str(t), indice_train_s)
                np.save(name_to_save + 'indice_test_s_' + str(t), indice_test_s)
                np.save(name_to_save + 'indice_train_ws_' + str(t), indice_train_ws)
                np.save(name_to_save + 'indice_test_ws_' + str(t), indice_test_ws)

            indice_train_s = np.load(name_to_save + 'indice_train_s_' + str(t) + '.npy').tolist()
            indice_test_s = np.load(name_to_save + 'indice_test_s_' + str(t) + '.npy').tolist()
            indice_train_ws = np.load(name_to_save + 'indice_train_ws_' + str(t) + '.npy').tolist()
            indice_test_ws = np.load(name_to_save + 'indice_test_ws_' + str(t) + '.npy').tolist()

    overlap = 1

    for I in ['I0', 'I1']:
        strI = I
        if I == 'I0':
            I = (1, 4)
            if dataset == 'Packer':
                I = (1, 8)
        if I == 'I1':
            I = (4, 9)
            if dataset == 'Packer':
                I = (8, 16)

        pairwise_label_probability = np.zeros((c, c))
        potentiel_labels = []
        for j in range(c):
            set_possible = list(set(torch.where(C[j] < I[1])[0].tolist()) & set(torch.where(C[j] >= I[0])[0].tolist()))
            if len(set_possible) > 12:
                potentiel_labels.append(random.sample(
                    list(set(torch.where(C[j] < I[1])[0].tolist()) & set(torch.where(C[j] >= I[0])[0].tolist())), k=12))

            else:
                potentiel_labels.append(set_possible + list(
                    random.sample(list(set(range(c)) - set(set_possible)), k=12 - len(set_possible))))
            # potentiel_labels.append(random.sample(list(set(torch.where(C[j] < I[1])[0].tolist())    & set(torch.where(C[j] >= I[0])[0].tolist())), k=12) )

        y_prior = []

        for yi in y:
            y_prior.append([yi] + random.sample(potentiel_labels[yi], k=9))

        X_s, X_ws, y_s, y_ws, y_s_prior, y_ws_prior = train_test_split(X, y, y_prior, stratify=y, test_size=0.5)

        print('Intersection label : ', len(set(y_s) & set(y_ws)), len(set(y)))


        name_to_save = path_file + str(overlap) + '_' + str(
            strI) + '_'
        if save:
            np.save(name_to_save + 'X_s', X_s)
            np.save(name_to_save + 'y_s', y_s)
            np.save(name_to_save + 'y_s_prior', y_s_prior)
            np.save(name_to_save + 'X_ws', X_ws),
            np.save(name_to_save + 'y_ws', y_ws)
            np.save(name_to_save + 'y_ws_prior', y_ws_prior)
            np.save(name_to_save + 'potentiel_labels', potentiel_labels)


        for t in range(5):
            X_train_s_0, X_test_s, y_train_0_s, y_test_s, indice_train_s, indice_test_s, y_train_0_s_prior, y_test_s_prior = train_test_split(
                torch.FloatTensor(X_s), y_s,
                range(len(X_s)), y_s_prior,
                test_size=0.33, stratify=y_s)
            X_train_ws_0, X_test_ws, y_train_0_ws, y_test_ws, indice_train_ws, indice_test_ws, y_train_0_ws_prior, y_test_ws_prior = train_test_split(
                torch.FloatTensor(X_ws), y_ws,
                range(len(X_ws)), y_ws_prior,
                test_size=0.33, stratify=y_ws)


            if save:
                np.save(name_to_save + 'indice_train_s_' + str(t), indice_train_s)
                np.save(name_to_save + 'indice_test_s_' + str(t), indice_test_s)
                np.save(name_to_save + 'indice_train_ws_' + str(t), indice_train_ws)
                np.save(name_to_save + 'indice_test_ws_' + str(t), indice_test_ws)

            indice_train_s = np.load(name_to_save + 'indice_train_s_' + str(t) + '.npy').tolist()
            indice_test_s = np.load(name_to_save + 'indice_test_s_' + str(t) + '.npy').tolist()
            indice_train_ws = np.load(name_to_save + 'indice_train_ws_' + str(t) + '.npy').tolist()
            indice_test_ws = np.load(name_to_save + 'indice_test_ws_' + str(t) + '.npy').tolist()



