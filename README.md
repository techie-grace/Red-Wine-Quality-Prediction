# Red-Wine-Quality-Prediction

## ABOUT
A simple machine learning project to predict the quality of red wine based on a dataset from kaggle (https://www.kaggle.com/vishalyo990/prediction-of-quality-of-wine/data).
The data contained 12 columns and 1599 rows. The quality of the wine was between 3 to 8, thus making it a multiclass classification problem.

## Data Analysis and Visualization
1. Import all the required libraries and read the dataset using pandas.
2. Get a description of the data by using the pandas `.head()` and `.describe()` methods. This allows you have a statistical summary of all the numerical columns involved.
3. Check for missing or null values using `.isna()` and `.sum()` methods on the dataframe. The output shows no missing values.
4. Using the python package, pandas profiling to carryout more data analysis and have a little more visualization of the interaction between the features. From the pandas profile report, we can observe that:
- The target variable is imbalanced.
- There are a few duplicated rows.
- Some of the features have outliers.

## Data Preprocessing
1. Encoded the categorical column that was engineered from the quality column using `LabelEncoder()` from `sklearn.preprocessing`
2. Using oversampling to deal with the problem of the imbalanced target data. `imblearn.over_sampling` has a class `RandomOverSampler` that does this.
3. Scaling the data using `StandardScaler` from `sklearn.preprocessing`

## Modelling
1. Random Forest classifier - gives an accuracy of 0.9
2. Decison Tree - gives an accuracy of 0.87 

## Reading a Multiclass Confusion Matrix
- TP - The true positive of a feature is the field where the actual and predicted value are the same.
- FN - The false negative of a feature is the sum of the values in the corresponding fields of that row except the TP.
- FP - The false positive of a feature is the sum of the values in the corresponding fields of that column except the TP.
- TN - The true negative of a feature is the sum of all values except the corresponding row and column of that feature.

## Further Work
Feature selection using gridsearch can be used to improve the models.
