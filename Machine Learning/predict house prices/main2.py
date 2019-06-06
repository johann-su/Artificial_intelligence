#!/Users/johann/anaconda3/bin/python

# importing the nessesary libarys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

# reading the data and making it usable
df = pd.read_csv("/Users/johann/github/Artificial_intelligence/Machine Learning/predict house prices/kc_house_data.csv")
df.head()

X = df.drop(["price", "date", "id", 'sqft_lot', 'sqft_lot15', 'yr_built', 'condition', 'zipcode'], axis=1).values
y = df[["price"]].values

# splitting in train & test data-sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.20)

model = joblib.load('Machine Learning/predict house prices/model_randomForests.pkl', 'r')

print(model.score(X_test, y_test))

# correlation matrix for all values in Dataframe
corr_matrix = df.corr()
corr_matrix["price"].sort_values(ascending=False)

atributes = ["price", "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "waterfront", "view", "condition", "grade", "sqft_above", "sqft_basement", "yr_built", "yr_renovated", "zipcode", "lat", "long", "sqft_living15"]
scatter_matrix(df[atributes], figsize=(12, 8))
