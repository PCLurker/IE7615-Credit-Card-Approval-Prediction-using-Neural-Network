# IE7615 - Credit Card Approval Prediction using Neural Network
This project implement an Artificial Neural Network (ANN) to solve the binary classification of credit card data as good credit or bad credit.

The data used comes from UC Irvine Machine Learning Repository: 
http://archive.ics.uci.edu/dataset/27/credit+approval


## Data Preprocessing:
This project is done entirely in Python. Libraries/packages used include NumPy, pandas, scikit-learn (sklearn), TensorFlow, 
Keras, and matplotlib, seaborn for visualization.

This process consists of the following steps:

* Check for missing values: There are 37 records with missing values. The missing values 
include both numeric and categorical values. We discard those records with any missing 
values. After this, our dataset has 653 records and 15 features, including both numerical 
and categorical variables. There are 296 bad records and 357 good records.
* Process categorical variables: All categorical variables are converted into new variables 
using one-hot encoding, resulting in a dataset with 46 features.
* Normalize data: A features’ values is normalized into (0, 1) range
* Check for variables correlation: Overall the data do not have correlated variables.
* Split the data into training, validation, and test set: We divide the dataset into 3 sets with a 
ratio of 70 : 15 : 15 for training, validation, and testing.

## Design model and training:
In this project, we implement an ANN model to train on the dataset.

The ANN consists of 3 layers: 

* Input layer with 46 units.
* 1 Dense hidden layer (have full connection with all units in the input layer) with 24 units,
with ReLU as activation function.
* Output layer with 2 units, corresponding to 2 prediction classes (good and bad credit), with 
SoftMax as activation function.

This ANN model is trained with the following configuration:

* Optimizer: Nadam (Adam with Nesterov momentum)
* Loss function: `keras.losses.SparseCategoricalCrossentropy`
* Batch size: 64
* Epoch: 64

## Evaluate the model:
The model are evaluated using accuracy metric.
