# Segmenting car parts with Deloitte
Final Project of Deep Learning at DTU.

This repo will hold the notebooks and scripts to train a neural network - **NOT** the actual data, as it is confidential. 
We are instead providing 10 manually generated and annotated images in the `data` directory, to replicate that of the given dataset. Despite there being a `test.npy` and `train.npy` in that directory, these are the same exact arrays and can be viewed as "placeholders". 
When evaluating the models with `evaluation.ipynb`, the performance is measured on these 10 test images. 

The structure of the repo is as follows:
```
.
├── data
│   ├── holds all data (10 manually generated images)
├── hpc
│   ├── holds the same structure as used during training with HPC
├── notebooks
│   ├── holds all the Jupyter notebooks (as remote)
└── src
    ├── holds all python source files (as remote)
```


***
 

To reproduce our results, first make sure that all libraries listed in `requirements.txt` are installed. Then, simply clone the repo and download the trained models by running the file `download_extract_models.py` from the root directory:
```bash
python3 src/download_extract_models.py
```
**Note:** the trained models require 965MB of free disk space on your computer. 

This will first download, then extract all the trained models to the folder `trained_models`, which is a requirement for the evaluation notebook to work. 
Finally, run the notebook called `evaluation.ipynb`, where you'll be able to evaluate each model's performance in the final cell, as well as viewing its prediction on one of the images from the test set. 