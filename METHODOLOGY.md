# Methodology Notes

## Objective
Predict `selling_price` from structured vehicle attributes and compare multiple modelling strategies under a common preprocessing setup.

## Split
- 80% training
- 20% testing
- random state 42

## Preprocessing
- StandardScaler for numerical variables
- OneHotEncoder for categorical variables
- unknown categories ignored during transformation

## Models
- Linear Regression
- Random Forest: 100 trees, max depth 10
- DNN: 64 -> 32 -> 1
- RF + PCA: five appended PCA components
- RF + K-Means: three-cluster label appended

## Evaluation
- MAE
- RMSE
- R²
- training time
- inference time

## Stored Experiment
The metrics and figures in `results/` are the completed outputs of the experiment. The notebook contains the corresponding training workflow for reproducibility.
