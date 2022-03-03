# bike-sharing-forecast
This repository contains all the pieces for a demo on Demand Forecast

The idea is to create a model that enable to predict the number of bicycles rented starting from daily and weather information.

## Original dataset
Can be found here: https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset

## Features
* LIghtGBM
* Oracle ADS
* ADSTuner for HPO
* catboost
* PyTorch TabNet
* K-fold CV
* Ensembling of different models (catboost, TabNet)

## Best result
The best result has been obtained doing an ensemble of the best results obtained with single models

