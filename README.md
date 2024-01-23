# Fair Tree-based Ensemble Regression Models
This repository contains a framework for building, measuring bias and correcting bias in gradient boosted trees (GBT), random forest (RF), and XGBoost  models using the XGBoost library. This takes advantage of the flexibility in XGBoost library to represent gradient boosted tree and random forest models, as well as the ability to use custom loss function.

## Testing for Bias
The [base_model_bias_testing.ipynb](https://github.com/NREL/Fair_Forest_Models/blob/main/base_model_bias_testing.ipynb) notebook is used to test for bias in a pre-trained machine learning XGBoost, GBT, and RF models. It takes as input:
* Pre-trained machine learning model.
* Test dataset to calculate the model's error.


Gamma Sweep Experiments
* These are the experiments that create the results_{model_type} folders
* You can edit the model types and corrections in run_gamma_sweeps.slurm to adjust which combinations are run
* To run, copy gamma_sweeps.py and run_gamma_sweeps.slurm to eagle and submit the slurm batch file
* Copy the resulting results_{model_type} folders to the local machine within the results folder of this repo
* Then you can generate the paper figures by running results_trends_gamma.ipynb

Note: the files are already on eagle under "/projects/mobility/ebensen/Tree_Based_Model_Bias/"
