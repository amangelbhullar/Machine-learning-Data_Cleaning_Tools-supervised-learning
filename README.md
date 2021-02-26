# Machine-learning-supervised-learning - python
# Object- Predicting customer purchase for a product based on salary, region(country), income
# Data preprocessing
# import liberaries and dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Dataset = pd.read_csv('Data.csv')
x = Dataset.iloc[:,:-1].values
y = Dataset.iloc[:,-1].values

# Taking care of missing data

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy='mean')
imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])

# Encoding categorical data (country & purchase status)

# Encoding the Independent Variable (country)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[0])], remainder='passthrough')

x = np.array(ct.fit_transform(x))

# Encoding the Dependent Variable (purchase)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(y)

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)

# Feature Scaling

