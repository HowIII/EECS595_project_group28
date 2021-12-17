# EECS595 project: Improving Verifiability of TRIP with Data Augmentation and Graph Neural Networks (Group28)
This repo is cloned from [Verifiable-Coherent-NLU](https://github.com/sled-group/Verifiable-Coherent-NLU) and modify some codes to add two new approaches.

## Getting Started
The data augmentation result can be reproduced using the jupyter notebook [TRIP_Data_Augmentation.ipynb](https://github.com/HowIII/EECS595_project_group28/blob/main/TRIP_Data_Augmentation.ipynb), which we ran in Colab with Python 3.7.  
### Incorporating LMs and GNNs
The conda virtual enviroment can be installed by the following commands:
```
conda env create -f trip_ours_env.yml
```
#### Graph construction and data preprocessing
__Data preparation__: To perform dependency parsing, we utilize the Stanford coreNLP dependency parser, and necessary files can be downloaded from https://drive.google.com/drive/folders/148bfSBczJhcHpgtPz98LA8am0MJAfTUW?usp=sharing (stanford-corenlp-4.2.2-models-english.jar and stanford-corenlp-4.2.2.zip). The ConceptNet Numberbatch embedding can also be downloaded from https://drive.google.com/drive/folders/148bfSBczJhcHpgtPz98LA8am0MJAfTUW?usp=sharing (numberbatch-en-19.08.txt).
As the graph construction process takes several hours, we have provided the preprocessed TRIP data in https://drive.google.com/drive/folders/148bfSBczJhcHpgtPz98LA8am0MJAfTUW?usp=sharing (tiered_dataset.pickle).
To preprocess the raw TRIP data, run the following commands:
```
python data_preprocessing_graph.py --drive_path DRIVE_PATH --save_pkl_path PATH_TO_SAVE_PICKLE_FILE --cn_nb_path numberbatch-en-19.08.txt --jar_path stanford-corenlp-4.2.2/stanford-corenlp-4.2.2.jar --models_jar_path stanford-corenlp-4.2.2-models-english.jar
```
The GNN result can be reproduced by the following commands:
```
python train_test_trip.py --drive_path DRIVE_PATH --pkl_file_path tiered_dataset.pickle --cn_nb_path numberbatch-en-19.08.txt
```

### Python Dependencies
The required dependencies for Colab are installed within the notebook, while the exhaustive list of dependencies for any setup is given in [requirements.txt](https://github.com/HowIII/EECS595_project_group28/blob/main/requirement.txt). Out of these, the minimal requirements can be installed in a new Anaconda environment by the following commands:
```
conda create --name tripPy python=3.7
conda activate tripPy
pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2
pip install transformers==4.2.2
pip install sentencepiece==0.1.96
pip install deberta==0.1.12
pip install spacy==3.2.0
python -m spacy download en_core_web_sm
pip install pandas==1.1.5
pip install matplotlib==3.5.0
pip install progressbar2==3.38.0
pip install ipykernel jupyter ipywidgets # For Jupyter Notebook setting
```

