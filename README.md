# IE7615 - Credit Card Approval Prediction using Neural Network

## Data Preprocessing:
This project is done entirely in Python. Our used libraries/packages include NumPy, pandas, scikit-learn (sklearn), TensorFlow, 
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

Our ANN consists of 3 layers: 

* Input layer with 46 units.
* 1 Dense hidden layer (have full connection with all units in the input layer) with 24 units,
with ReLU as activation function.
* Output layer with 2 units, corresponding to 2 prediction classes (good and bad credit), with 
SoftMax as activation function.

This ANN model is trained with the following configuration:

* Optimizer: Nadam (Adam with Nesterov momentum)
* Loss function: keras.losses.SparseCategoricalCrossentropy
* Batch size: 64
* Epoch: 64
* 
Support Vector Machine (SVM) is a supervised machine learning algorithm. SVM can be used for 
classification and regression problems. SVM plots each data item as a point in dimensional space. The 
value of each feature belongs to a particular coordinate. Classification was performed by finding 
hyperplanes that divide two classes.

If the separator for the dataset is not linear, a non-linear SVM model can be used instead, which 
will transform the data into higher dimension space, where a linear hyperplane can be found.

## Evaluate the model:
We evaluate all models using accuracy metric.
