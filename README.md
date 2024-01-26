# Fair Tree-based Ensemble Regression Models
This repository contains a framework for building, measuring bias and correcting bias in gradient boosted trees (GBT), random forest (RF), and XGBoost  models using the XGBoost library. This takes advantage of the flexibility in XGBoost library to represent gradient boosted tree and random forest models, as well as the ability to use custom loss function.

## Testing for Bias
The [base_model_bias_testing.ipynb](https://github.com/NREL/Fair_Forest_Models/blob/main/base_model_bias_testing.ipynb) notebook is used to test for bias in a pre-trained XGBoost, GBT, and RF models. This approach can also be used to test bias in other machine learning models that can be implemented in this framework. The bias testing method takes as input:
* Pre-trained machine learning model.
* Test dataset to calculate the model's error.

## Bias Mitigation
To do bias mitigation, we train the tree-based models with various gamma values to understand the trade-off between model performance and demographic bias in the models. This allows the user to create custom models with their specified tolorance for bias and model performance. We use the gamma_sweeps.py to do a thorough analysis of the models types and the correction terms. See below for more instructions.


### Script Parameters
Model Type
| Parameter | Description |
| ----------- | ----------- |
| rf | Random Forest Model|
| gbt | Gradient Boosted Trees |
| xgb | Extreme Gradient Boosted Trees |

Correction Terms
| Parameter | Description |
| ----------- | ----------- |
| pearson | Pearson correlation for linear relationship|
| distance | Distance correlation for non-linear |
| kendall | Kendall's Tau for non-linear. This will run the slowest |

```linux
python -W ignore gamma_sweeps.py --model_type {model type} --correction {correction term}
```
* These are the experiments that create the results_{model_type} folders
* To run gamma_sweeps.py in terminal:

  
should look like this if code is running proper:
```{r, message=TRUE}
Model: rf, Correction: distance, Loading data...
Model: rf, Correction: distance, Sweeping Gamma...
Model: rf, Correction: distance, Gamma: 0.5, 1/51
[10:44:59] WARNING: /var/folders/nz/j6p8yfhx1mv_0grj5xl4650h0000gp/T/abs_21wtzqx5vy/croot/xgboost-split_1675457780668/work/src/learner.cc:767: 
Parameters: { "num_parallel_trees" } are not used.


```
* Results will populate in  the results_{model_type} folders to the local machine within the results folder of this repo
* Then you can generate the paper figures by running results_trends_gamma.ipynb

