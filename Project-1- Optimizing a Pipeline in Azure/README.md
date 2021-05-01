# Optimizing an ML Pipeline in Azure

## Table of contents
* [Overview](#Overview)
* [Summary](#Summary)
* [Scikit-learn Pipeline](#Scikit-learn-Pipeline)
* [AutoML](#AutoML)
* [Pipeline comparison](#Pipeline-comparison)
* [Future work](#Future-work)
* [Proof of cluster clean up](#Proof-of-cluster-clean-up)
* [Citation](#Citation)
* [References](#References)

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a Scikit-learn Logistic Regression model. Two approaches were implemented and tested, the first was using the SKLearn Logistic Regression and the hyperparameters were tuned using the Hyperdrive, and the second implementation was using AutoML. The results from both results were compared to each other. <br>
The diagram below shows the full overview of the project.
![Diagram](images/project_overview.png?raw=true "Main Steps of the Project")


## Summary
The dataset provided contains data related to a direct marketing campaign for a bank. The goal is to predict if a client will subsrcibe a term deposit or not (i.e a binary classification). There are 20 features (columns) in total which includes customers age, job, maritual...etc. and the target column "y" which is a "Yes" or "No" that needs to be encoded to 0 and 1. The original dataset can be found [here](https://archive.ics.uci.edu/ml/datasets/bank+marketing). <br> <br>
The best performaing model was the VotingEnsemble from the AutoML with an accuracy of 0.91709 which is slightly higher than the logistic regression (tuned with Hyperdrive) which had and accuracy of 0.91.

## Scikit-learn Pipeline
For the Hyperdrive to tune the parameters, the parameter sampler had to be defined first. For the logistic regression, the two parameters were defined in the parameter sample as follows:
```
ps = RandomParameterSampling({
    "--C": uniform(0.03, 1),
    "--max_iter": choice(50, 100, 150, 200, 300)
})
```
There are three choices for the sampling methods: Random sampling, grid sampling, and Bayesian sampling. The grid sampling is the most expensive one as its an exhaustive search over the hyperparameter space. Bayesian sampling is baswed on Bayesian optimization algorithm and similar to Grid sampling, it is recommended if we have enough budget to explore the hyperparamter space. The search space can a discrete or continous. In the bove code snippet, the **choice** specific discrete values search and **uniform** specifies continous hyperparameters. More information regardin the parameter sample can be found in the [Azure documentation](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters).

![image](https://user-images.githubusercontent.com/43079200/116450659-ef80ed80-a831-11eb-8b89-d94c1750dccf.png)

![image](https://user-images.githubusercontent.com/43079200/116450716-00c9fa00-a832-11eb-9e95-5bdc8d96bc4e.png)

[C=0.14533, max_iter=150]

## AutoML


## Pipeline comparison


## Future work


## Proof of cluster clean up

