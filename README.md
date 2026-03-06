# BioHSNet
Drug Target Prediction from Perturbation Transcriptomics via a Biological Function-Guided Hypergraph Siamese Network
## Installation
### 1. Environment
Python 3.8
### 2. setup
```bash
cd BioHSNet
conda env create -n SSGCN -f pytorch_zxy.yaml
```
## Dataset
The level 5 perturbation profiles from the L1000 CMap dataset were downloaded from the Gene Expression Omnibus (GEO) database (accession numbers: GSE92742 and GSE70138) via [Google Drive](https://www.ncbi.nlm.nih.gov/geo).Gene Ontology (GO) Data: The human Gene Ontology data was downloaded from [Google Drive](https://current.geneontology.org).
### Data Preprocessing
We provide a preprocessing script at `code/data_preprocessing.py` to prepare the raw data for training and evaluation.
## Training (Example)
After preprocessing the data, you can train BioHSNet using `code/model/train.py`.
### Run
```bash
python code/model/train.py
```
