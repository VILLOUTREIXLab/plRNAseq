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

4) For average results :  
```
python3 -u pl_main_average.py
```
5) For Gridsearch hyperparameters : Note you may need computation ressources, in this case you may change few thing in code see next paragraph  
```
python3 -u pl_main_average_gridsearch.py
```
6) Need to change PATH file and save file
7) missing things : Prostt data, load  and creation 
   
