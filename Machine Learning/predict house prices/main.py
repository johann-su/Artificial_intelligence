#!/Users/johann/anaconda3/bin/python

# importing the nessesary libarys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix

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

# normalizing the dataframe
#df_norm = (df - df.mean()) / (df.max() - df.min())

X = df.drop(["price", "date", "id", 'sqft_lot', 'sqft_lot15', 'yr_built', 'condition', 'zipcode'], axis=1).values
y = df[["price"]].values

#small portion of the dataframe to speed up Grid Search
X_small = X[:5000]
y_small = y[:5000]

# splitting in train & test data-sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.20)

# specifiing Hyperparameters for GridSearchCV
param_grid = [{'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'gamma': [1, 2, 3, 4, 5, 6], 'max_iter': [-1, 2, 4, 6, 8, 10, 12]}]

# scaling the data
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# defining a model
model = SVR(gamma=1, kernel='poly', max_iter=-1)
model.fit(X_train, y_train)

# performimg a GridSearch to find the best Hyperparameters
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

#print(grid_search.best_params_)
print(model.score(X_test, y_test))
print(model.score(X_train, y_train))

# saving/loading the model
joblib.dump(model, "Machine Learning/predict house prices/model_svm.pkl")
