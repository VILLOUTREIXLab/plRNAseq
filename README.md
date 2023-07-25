# plRNAseq
partial label learning for single cell RNA seq data classification. This code was produced for this article : biorxiv link  

![figure1.jpg](figure1.jpg)


0) Download  
```
git clone https://github.com/MalekSnous/plRNAseq.git
```   
1) create virtual environment :
```
cd plRNAseq
python3 -m pip install --user --upgrade pip
python3 -m venv plenv
source plenv/bin/activate
pip install -r pl_requirements.txt
```
you can desactivate the virtual environment with :  
```  
deactivate
```


2) unzip data in the current repertory :  

```
mkdir -p data/datasets/  
unzip Packer.zip -d data/datasets/  
unzip Planaria.zip -d data/datasets/  
unzip Paul.zip -d data/datasets/  
```

3) run main script:  
```
python3 -u pl_main.py  
```

4) For average results :  note you may need to search good hyperparameters ...
```
python3 -u pl_main_average.py
```
5) For Gridsearch hyperparameters : Note you may need computational ressources, in this case you may change few thing in code see next paragraph  
```
python3 -u pl_main_average_gridsearch.py
```

6) In addition, if you want to use partial lebelling, you can use script :
```
python3 -u create_partial_label_ppp.py
```
8) For prosstt dataset, in order to run , you need to downland some additional packages and create_da:
```
pip install git+git://github.com/soedinglab/prosstt.git
jupyter lab  create_tree.ipynb
python3 -u create_partial_label_prosstt.py
```
   
