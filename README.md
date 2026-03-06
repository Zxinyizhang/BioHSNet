# BioHSNet
Drug Target Prediction from Perturbation Transcriptomics via a Biological Function-Guided Hypergraph Siamese Network
## Installation
### 1. Environment
Python 3.8
### 2. setup
```bash
cd SSGCN
conda env create -n SSGCN -f pytorch_zxy.yaml
```bash
## Dataset
The level 5 perturbation profiles from the L1000 CMap \cite{subramanian2017} dataset were downloaded from the Gene Expression Omnibus (GEO) database (accession numbers: GSE92742 and GSE70138) via \url{https://www.ncbi.nlm.nih.gov/geo}.Gene Ontology (GO) Data: The human Gene Ontology data was downloaded from \url{https://current.geneontology.org}.
