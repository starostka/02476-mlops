MLops pipeline for Classification of Scietific Papers
==============================
![example workflow](https://github.com/Starostka/02476-mlops/actions/workflows/branch-push.yml/badge.svg)
![example workflow](https://github.com/Starostka/02476-mlops/actions/workflows/main-pull-request.yml/badge.svg)
![example workflow](https://github.com/Starostka/02476-mlops/actions/workflows/isort.yml/badge.svg)
<br/>
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
![Google Cloud](https://img.shields.io/badge/GoogleCloud-%234285F4.svg?style=for-the-badge&logo=google-cloud&logoColor=white)

This repository was made as a placeholder for the final project of DTU course Machine Learning Operations (02476). The scope of the project was to deploy an ML pipeline on the cloud. The model used in this project was made using Pytorch Geometric package and the goal was to classify Scientific papers.
<br/>
<br/>
Contributors: Jens Perregaard Thorsen, Benjamin Starostka Jakobsen, Philippe Gonzalez and Spyros Vlachospyros

# Project Description

## Overall goal of the project
The goal of the project is to apply a geometric model and train/deploy the model in the cloud. The main focus of the project will be around the frameworks and the tools needed to do that and not in the model performance.

## What framework are you going to use (Kornia, Transformer, Pytorch-Geometrics)
We’ll be using the Pytorch Geometric (Pyg) framework.
How to you intend to include the framework into your project
The framework already contains a collection of state-of-the-art models for dealing with unstructured/graph data. Thus, we intend to start using one of the existing models in predicting tasks once we have learned more about the dataset and the problems we wish to solve.

## What data are you going to run on (initially, may change)
Our initial project idea is to use PyTorch Geometric models and the CORA dataset to classify scientific papers based on their content. The CORA dataset consists of more than 2,500 scientific papers from seven different categories, and includes both the paper text and the citation graph between the papers.

## What deep learning models do you expect to use
To classify the scientific papers, we will try to use a graph convolutional neural network (GCN) or a graph attention network (GAT). These models take a graph as input and learn to classify the nodes (i.e., the papers) based on their features and connectivity.

## Revisit of our initial project description
After finalizing our project and seeing our initial project description and goal we all agreed that the project turned out even better than we had initially envisioned. All the goals and objectives were met or at least we tried to meet them at a substantial level. The final outcome met most of the requirements and specifications and the team did an excellent job in executing the initial project plan. Our initial decision of focusing on the pipeline and on all the important modules of the course paid off since we had more time to focus on it. Overall, the course did teach us a lot of valid staff that we will be able to use later in our academic or work careers.  

## WandB report:
See the following overview report of the model performance: [Overview](https://wandb.ai/02476-mlops-12/Pytorch%20Geometric%20Model/reports/Model-Report--VmlldzozMzYyMjY0)

# Contribute
How do we go about it? Read the checklist -> branch out -> fix the task -> create pull request.


## Practicalities

Configure environment:

    # with existing environment activated:
    pip install -r requirements.txt

Download and make the dataset:
```
python src/data/make_dataset.py
```

Train the model:
```
python src/models/train.py
```

Test the model:
```
python src/models/evaluate.py
```

Make a single prediction:
```
python src/models/predict.py <item-index-in-dataset>
```

Run unittests with coverage and get report:
```
coverage run --source=src/ -m pytest tests/
coverage report
```

Submit training job to Vertex AI:
```
gcloud ai custom-jobs create --region=europe-west1 --display-name=training_job --config=vertex_jobspec.yaml
```

Create updated data-drift report
```
python src/models/monitoring.py
```

### Configure Torch on M1 (FIX)
Create a new conda environment using the locked environment:
```
conda env create -n mlops --file environment-m1.yml
```
if it still fails.. Run the `utilities/conda-torch-m1.sh` shell script in a fresh conda environment. And then continue to install your packages as usual.
- Remember to make the script executable i.e., `chmod +x utilities/conda-torch-m1.sh`..

### Build documentation
The documentation is built using Sphinx:
```
sphinx-build docs _build
```
and deployed to GitHub Pages using a workflow on commits to the `gh-pages` branch. 

## Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the project and running it locally
    ├── requirements-docker.txt   <- The requirements file for running the docker file as some packages are installed individually
    │                         in the Dockerfile
    ├── requirements-dev.txt      <- The requirements file containing additional packages for project development
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │  
    │   ├── config         <- Experiment configuration files to be used with hydra
    │   │   ├── experiment <- Various expriment setups
    │   │   │   └── exp1.yaml
    │   │   └── default_config.yaml
    │   │  
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to define arcitectur, train models, use trained models to make
    │   │   │                 predictions and for cprofiling the model scripts
    │   │   ├── predict_model.py
    │   │   ├── model.py
    │   │   ├── train_model_cprofile_basic.py
    │   │   ├── train_model_cprofile_sampling.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
