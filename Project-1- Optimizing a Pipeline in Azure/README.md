# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
The dataset provided contains data related to a direct marketing campaign for a bank. The goal is to predict if a client will subsrcibe a term deposit or not (i.e a binary classification). There are 20 features (columns) in total which includes customers age, job, maritual...etc. <br> <br>
Two approaches were implemented and tested, the first was using a logistic regression from sklearn and the hyperparameters were tuned using the Hyperdrive, and the second implementation was using AutoML. <br>
The best performaing model was the VotingEnsemble from the AutoML with an accuracy of 0.91709 which is slightly higher than the logistic regression (tuned with Hyperdrive) which had and accuracy of 0.91.

## Scikit-learn Pipeline

![image](https://user-images.githubusercontent.com/43079200/116450659-ef80ed80-a831-11eb-8b89-d94c1750dccf.png)

![image](https://user-images.githubusercontent.com/43079200/116450716-00c9fa00-a832-11eb-9e95-5bdc8d96bc4e.png)


## AutoML


## Pipeline comparison


## Future work


## Proof of cluster clean up

