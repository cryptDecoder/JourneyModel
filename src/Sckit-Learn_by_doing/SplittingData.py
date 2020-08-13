# standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# getting our data ready
car_sales = pd.read_csv("../../data/datasets_1211_2177_car_ad.csv", encoding="ISO-8859-1")
print(car_sales.head())
print(len(car_sales))

# split yhe data into X and y

from sklearn.model_selection import train_test_split

print("Getting our data ready")
X = car_sales.drop('price', axis=1)
y = car_sales['price']

print("X data:", X)
print("Y data :", y)

print("Splitting our data")
# split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Build machine learning model
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# finding missing data from dataset


# let's try convert missing data into numbers values
# create X and y again

X = car_sales.drop('price', axis=1)
y = car_sales['price']

print("Showing missing data details :", car_sales.isna().sum())
# fill missing data with pandas

print("Total records before remove missing values :", len(car_sales))
# remove rows with missing values
car_sales.dropna(inplace=True)
print("Total records after remove missing values", len(car_sales))
print("Showing missing data details after filna :", car_sales.isna().sum())

X = car_sales.drop('price', axis=1)
y = car_sales['price']

print("Build machine learning model")
clf = RandomForestClassifier()
clf.get_params()

categorical_featurs = ["car", "body", "engType", "registration", "model", "drive"]
one_hot = OneHotEncoder()

transformer = ColumnTransformer([('encoder', one_hot, categorical_featurs)], remainder='passthrough')

transformed_X = transformer.fit_transform(X)
print(transformed_X)

# fit the model
np.random.seed(42)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

print(y)
X_train, X_test, y_train, y_test = train_test_split(transformed_X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))


