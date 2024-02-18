import torch
import torch.nn as nn


from pl_model import pl_nn_prototybe_based, pl_hKNN, plSVM, pl_KNN, plRF,  plkernelSVC, plXGBM, plEMLR, plEMSVM, plLR




# liste_method = [

# # ALGO IRL 
# 'PB',  # Prototype Based
# 'plSVM',  # Support Vector Machine  
# 'plLR',  # Logistic Regression

# # Special

# 'plhKNN',  # k Nearest Neighbors

# # Algo IFR 
# 'plRF', # Random Forest
# 'plXGBM', # Extreme Gradient Boosting Method
# 'plkernelSVC', # kernel SVM (real kernel implmentation)
# 'plEMLR',   # LR with algo IFR
# 'plEMSVM', # SVM with algo IFR
# ]

def warn(*args, **kwargs):
    pass



import warnings
warnings.filterwarnings("ignore")

def init_dict_0(method, indice_network, dim_int, c, mat_C, C, device, flat=True, developpement=False, supervised=False):
    if method == 'plSVM' :
        
        epochs = 5 if developpement else 101
        kernel = True if indice_network == 2  else False
        dict_0 = {'W':nn.Sequential(nn.Linear(dim_int, c)),  'optimizer':'Adam', 'lr_P':1.e-5,  'epochs':epochs, 'device':device,  'C':mat_C,  'C_score':C ,'kernel':kernel,  'dim_int':dim_int, 'c':c}
        tiny_model =  plSVM(W=nn.Sequential(nn.Linear(dim_int, c)),kernel=kernel)

    if method == 'plLR' :
        
        epochs = 5 if developpement else 201
        kernel = True if indice_network == 2  else False
        dict_0 = {'W':nn.Sequential(nn.Linear(dim_int, c), nn.Softmax(dim=1)),  'optimizer':'Adam', 'lr_P':5.e-4,  'epochs':epochs, 'device':device,  'C':mat_C,  'C_score':C ,'kernel':kernel, 'dim_int':dim_int, 'c':c}
        tiny_model =  plLR(W=nn.Sequential(nn.Linear(dim_int, c),  nn.Softmax(dim=1)))


    if method == 'plEMSVM' :
        
        epochs = 5 if developpement else 101
        kernel = True if indice_network == 2  else False
        dict_0 = {'W':nn.Sequential(nn.Linear(dim_int, c)),  'optimizer':'Adam', 'lr_P':1.e-5,  'epochs':epochs, 'device':device,  'C':mat_C,  'C_score':C ,'kernel':kernel, 'dim_int':dim_int, 'c':c, 'supervised':supervised}
        tiny_model =  plEMSVM(W=nn.Sequential(nn.Linear(dim_int, c)),)


    if method == 'plEMLR' :
        dict_0  = {'C':mat_C,  'C_score':C , 'epochs':10,  'supervised':supervised, 'W':nn.Sequential(nn.Linear(dim_int, c), nn.Softmax(dim=1))}
        tiny_model = plEMLR(W=nn.Sequential(nn.Linear(dim_int, c),  nn.Softmax(dim=1)))


        

    if method == 'plhKNN' :
        dict_0  = {'C':mat_C,  'C_score':C , 'flat':flat}
        tiny_model = pl_hKNN()


    if method in ['plRF', 'plRF_check','pl2RF'] :
        dict_0  = {'C':mat_C,  'C_score':C , 'epochs':10, 'supervised':supervised}
        tiny_model = plRF()


    if method == 'plXGBM' :
        dict_0  = {'C':mat_C,  'C_score':C , 'epochs':10,  'supervised':supervised}
        tiny_model = plXGBM( 

                # param PL algo
                epochs=5,
                supervised=False,


                # param for metric and computation
                indice_network =0,
                device='cpu',
             
                C=torch.ones(26),
                C_score=torch.ones(26),)

    if method in [ 'plkernelSVC','plkSVC'] :
        dict_0  = {'C':mat_C,  'C_score':C , 'epochs':10,  'supervised':supervised}
        tiny_model = plkernelSVC()

  

    if method == 'PB' :
       

        epochs_regression, epoch_xsi = 31,31
        if developpement : 
            epochs_regression, epoch_xsi = 5,5
        g = dim_int
        liste_nn = [
    nn.Sequential( nn.Linear(g,c)),
    nn.Sequential( nn.Linear(g,c), nn.Tanh(), nn.Linear(c,c)),
    nn.Sequential( nn.Linear(g,c), nn.Tanh(), nn.Linear(c,c), nn.Tanh(), nn.Linear(c,c)),
    nn.Sequential( nn.Linear(g,c), nn.Tanh(), nn.Linear(c,c), nn.Tanh(), nn.Linear(c,c), nn.Tanh(),  nn.Linear(c, c)),
]
        P = liste_nn[indice_network]

        dict_0  = {'Network':P,  'optimizer':'Adam',  'lr_P':1.e-5,
               'epochs_regression':epochs_regression,    'epochs_xsi':epoch_xsi,
          'device':device,          'C':mat_C,          'C_score':C , 'dim_int':dim_int, 'c':c}
        tiny_model = pl_nn_prototybe_based()
    return dict_0, tiny_model 

def init_method(method, dict_entry):
    if method == 'plSVM' :    
        #print(dict_entry)         
        model = plSVM(W=nn.Sequential(nn.Linear(dict_entry['dim_int'], dict_entry['c'])),
                            C = dict_entry['C'],
                            C_score = dict_entry['C_score'],
                              device = dict_entry['device'],
                              lambdaa = dict_entry['lambdaa'],
                              epochs = dict_entry['epochs'],
                              kernel = dict_entry['kernel'],
                              p = dict_entry['p'],
                              gamma = dict_entry['gamma'],


                              )


    if method == 'plEMSVM' :             
        model = plEMSVM(W=nn.Sequential(nn.Linear(dict_entry['dim_int'], dict_entry['c'])),
                            C = dict_entry['C'],
                            C_score = dict_entry['C_score'],
                              device = dict_entry['device'],
                              lambdaa = dict_entry['lambdaa'],
                              epochs = dict_entry['epochs'],
                              kernel = dict_entry['kernel'],
                             supervised=dict_entry['supervised']
                              )

    if method == 'plLR' :
          model = plLR(W=nn.Sequential(nn.Linear(dict_entry['dim_int'], dict_entry['c']), nn.Softmax(dim=1)),
                            C = dict_entry['C'],
                            C_score = dict_entry['C_score'],
                              device = dict_entry['device'],
                              lambdaa = dict_entry['lambdaa'],
                              epochs = dict_entry['epochs'],
                              kernel = dict_entry['kernel'],)


    if method == 'plhKNN' :
        model = pl_hKNN(
                        k = dict_entry['k'],
                        C = dict_entry['C'],
                        C_score = dict_entry['C_score'],
                        flat = dict_entry['flat']
                       
                      )


    if method == 'PB' :
        model = pl_nn_prototybe_based(Network = dict_entry['Network'],
                                      lambdaa = dict_entry['lambdaa'],
                                      lambdaa_solution_regression = dict_entry['lambdaa_solution_regression'],
                                      optimizer = dict_entry['optimizer'],
                                      distance = dict_entry['distance'],
                                      lr_P = dict_entry['lr_P'],
                                      epochs_regression = dict_entry['epochs_regression'],
                                    epochs_xsi = dict_entry['epochs_xsi'],
                                      device = dict_entry['device'],
                                      C = dict_entry['C'],
                                      C_score = dict_entry['C_score'])

    if method in ['plRF', 'plRF_check','pl2RF']:
        model = plRF(   n_estimators=dict_entry['n_estimators'],
                        max_depht =dict_entry['max_depht'],
                        epochs=dict_entry['epochs'],
                        indice_network =0,

                        #device=dict_entry['device'],
                        max_features=dict_entry['max_features'],
                        criterion=dict_entry['criterion'],
                        C = dict_entry['C'],
                        C_score = dict_entry['C_score'],
                        supervised=dict_entry['supervised']
                                      )



    if method == 'plXGBM' :
         model = plXGBM(    n_estimators=dict_entry['n_estimators'],
                            max_depht =dict_entry['max_depht'],
                   

                            learning_rate = dict_entry['learning_rate'],
                            gamma = dict_entry['gamma'],
                            min_child_weight= dict_entry['min_child_weight'],


                                epochs=dict_entry['epochs'],
                                indice_network =0,
                               



                    C = dict_entry['C'],
                    C_score = dict_entry['C_score'],
                     supervised=dict_entry['supervised']
                                  )


    if method in [ 'plkernelSVC','plkSVC'] :
        model = plkernelSVC(  lambdaa=dict_entry['lambdaa'],
                    epochs=dict_entry['epochs'],
                    indice_network =2,
                    gamma=dict_entry['gamma'],
                    C = dict_entry['C'],
                    C_score = dict_entry['C_score'],
                     supervised=dict_entry['supervised']
                                  )

    if method == 'plEMLR' :
        model = plEMLR(  W=dict_entry['W'],
                    lambdaa=dict_entry['lambdaa'],
                    epochs=dict_entry['epochs'],
                #     tol=dict_entry['tol'],
                # penalty=dict_entry['penalty'], 
                # solver=dict_entry['solver'], 
                    
                        C = dict_entry['C'],
                            C_score = dict_entry['C_score'],
                             
                           
                        
                           

                
                                  )




    return model 
