# standard import all required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#  Things to be remember

'''
    All data should be numeric
    There should be no missing values
    Manipulate the test set the same as the training set
    Never test on data you've trained on
    Tune hyperparameters on validation set or use cross-validation
    One best performance metric doesn't mean the best model
'''

# Scikit learn pipeline
# 1: Getting data ready


# Step we want to do (all in one cell)
'''
    1: Fill missing values
    2: Convert data ti numbers
    3: Build a model on the data
'''

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Modelling

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV

# setup random seed
np.random.seed(42)

# import data and drop rows with missing labels


data = pd.read_csv('../../data/car-sales-extended-missing-data.csv')
print(data.head())
print(data.isna().sum())
data.dropna(subset=['Price'], inplace=True)

# define different features and transformer pipeline

categorical_features = ['Make', 'Colour']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy="constant", fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

door_feature = ['Doors']
door_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy="constant", fill_value=4))
])

numeric_features = ['Odometer (KM)']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])
# setup the preprocessing data
preprocessor = ColumnTransformer(
    transformers=[(
        'cat', categorical_transformer, categorical_features),
        ('door', door_transformer, door_feature),
        ('num', numeric_transformer, numeric_features)
    ]
)

# creating and modelling pipeline
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('model', RandomForestRegressor())])

# Split the data
X = data.drop('Price', axis=1)
y = data['Price']

print("Printing the value of X and y")
print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# fit and score the model

model.fit(X_train, y_train)
print(model.score(X_test, y_test))

# Use GridSearchCV with our regression pipeline


pipe_grid = {
    "preprocessor__num__imputer__strategy": ['mean', 'median'],
    'model__n_estimators': [100, 1000],
    'model__max_depth': [None, 5],
    'model__max_features': ['auto'],
    'model__min_samples_split': [2, 4]
}
gs_model = GridSearchCV(model, pipe_grid, cv=5, verbose=2)
gs_model.fit(X_train, y_train)

# evalute model
print(gs_model.score(X_test, y_test))
