# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
The dataset provided contains data related to a direct marketing campaign for a bank. The goal is to predict if a client will subsrcibe a term deposit or not (i.e a binary classification). There are 20 features (columns) in total which includes customers age, job, maritual...etc. <br> <br>
Two approaches were implemented and tested, the first was using a logistic regression from sklearn and the hyperparameters were tuned using the Hyperdrive, and the second implementation was using AutoML. <br>
The best performaing model was the VotingEnsemble from the AutoML with an accuracy of 0.91709. 

## Scikit-learn Pipeline



## AutoML


## Pipeline comparison


## Future work


## Proof of cluster clean up

