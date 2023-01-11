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

# Contribute
How do we go about it? Read the checklist -> branch out -> fix the task -> create pull request.


## Practicalities

Configure environment:

    conda create --name mlops --file requirements.txt

    # with existing environment activated:
    conda install --file requirements.txt

    # if packages are not available from current channels add conda-forge channel:
    conda config --append channels conda-forge

Download and make the dataset:
```
python src/data/make_dataset.py data/processed/
```

Train the model:
```
python src/models/train_model.py
```

Test the model:
```
python src/models/predict_model.py models/trained_model.pt
```

Run unittests with coverage
```
coverage run --source=src/ -m pytest tests/
```


# Project checklist

## Week 1

-   [X] Create a git repository
-   [X] Make sure that all team members have write access to the github repository
-   [ ] Create a dedicated environment for you project to keep track of your packages (using conda)
-   [X] Create the initial file structure using cookiecutter
-   [X] Fill out the \`make<sub>dataset.py</sub>\` file such that it downloads whatever data you need and
-   [X] Add a model file and a training script and get that running
-   [X] Remember to fill out the \`requirements.txt\` file with whatever dependencies that you are using
-   [X] Remember to comply with good coding practices (\`pep8\`) while doing the project
-   [ ] Do a bit of code typing and remember to document essential parts of your code
-   [ ] Setup version control for your data or part of your data
-   [X] Construct one or multiple docker files for your code
-   [X] Build the docker files locally and make sure they work as intended
-   [ ] Write one or multiple configurations files for your experiments
-   [ ] Used Hydra to load the configurations and manage your hyperparameters
-   [ ] When you have something that works somewhat, remember at some point to to some profiling and see if
    you can optimize your code
-   [ ] Use wandb to log training progress and other important metrics/artifacts in your code
-   [ ] Use pytorch-lightning (if applicable) to reduce the amount of boilerplate in your code


<a id="org59b93c8"></a>

## Week 2

-   [ ] Write unit tests related to the data part of your code
-   [ ] Write unit tests related to model construction
-   [ ] Calculate the coverage.
-   [ ] Get some continuous integration running on the github repository
-   [ ] (optional) Create a new project on \`gcp\` and invite all group members to it
-   [ ] Create a data storage on \`gcp\` for you data
-   [ ] Create a trigger workflow for automatically building your docker images
-   [ ] Get your model training on \`gcp\`
-   [ ] Play around with distributed data loading
-   [ ] (optional) Play around with distributed model training
-   [ ] Play around with quantization and compilation for you trained models


<a id="orgf0bbc13"></a>

## Week 3

-   [ ] Deployed your model locally using TorchServe
-   [ ] Checked how robust your model is towards data drifting
-   [ ] Deployed your model using \`gcp\`
-   [ ] Monitored the system of your deployed model
-   [ ] Monitored the performance of your deployed model


<a id="org8196375"></a>

## Additional

-   [ ] Revisit your initial project description. Did the project turn out as you wanted?
-   [ ] Make sure all group members have a understanding about all parts of the project
-   [ ] Create a presentation explaining your project
-   [ ] Uploaded all your code to github
-   [ ] (extra) Implemented pre\*commit hooks for your project repository
-   [ ] (extra) Used Optuna to run hyperparameter optimization on your model


<a id="orge6c5843"></a>

# Project package list

Heres a small list of packages that could be usefull during this project,

-   pre-commit, to automatically run black, mypy, nbdev etc.. on commits
-   conda environment
-   nbdev, to avoid notebook merge conflicts
-   dvc, vc for large files
-   flake8, check code according to pep8
-   black, fix code formatting according to pep
-   isort, sort imports
-   mypy, static type checker
-   pipreqs, generate python requirements
