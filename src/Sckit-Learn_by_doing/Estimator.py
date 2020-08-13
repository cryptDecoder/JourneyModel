# import Boston housing dataset
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np

boston = load_boston()
print(boston)

boston_df = pd.DataFrame(boston['data'], columns=boston['feature_names'])
boston_df['target'] = pd.Series(boston['target'])
print(boston_df.head())

# check how many samples
print(len(boston_df))

# lets try ridge rigression model
from sklearn.linear_model import Ridge

# setup random seed

np.random.seed(42)
X = boston_df.drop('target', axis=1)
y = boston_df['target']

# split data into test train
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_ytest = train_test_split(X, y, test_size=0.2)

# instantiate Ridge model
model = Ridge()
model.fit(X_train, y_train)

# check the score of the model
print(model.score(X_test, y_ytest))

# how  do we improve this score
# what if ridge is wasn't working

# lets try Randomforest regression

print(" Working with RandoForest Regressor")
from sklearn.ensemble import RandomForestRegressor

# setup random seed

np.random.seed(42)
X = boston_df.drop('target', axis=1)
y = boston_df['target']

# split data into test train
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# instantiate Random forest regressor
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train, y_train)

# evaluate model
print(rf.score(X_test, y_test))

# choosing estimator for classification problem
print("Working with classification")
heart_disease = pd.read_csv('../../data/datasets_33180_43520_heart.csv')
print(heart_disease.head())
print(len(heart_disease))
from sklearn.svm import LinearSVC

np.random.seed(42)
X = heart_disease.drop('target', axis=1)
y = heart_disease['target']
# split the data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2)

# Instantiate LinearSVC
clf = LinearSVC()
clf.fit(X_train, y_train)
# Evaluate MOdel
print(clf.score(X_test, y_test))

# Working with random forest classifier

from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)
X = heart_disease.drop('target', axis=1)
y = heart_disease['target']
# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

# Fit the model and use it to make predication the data
# fiiting the model the data
print("Fitting the model the data")

from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)
X = heart_disease.drop('target', axis=1)  # features variables data
y = heart_disease['target']  # labels, targets , target variables
# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2)

clf = RandomForestClassifier()

# Fit the model / training the machine learning model
clf.fit(X_train, y_train)
# Evaluate the machine learning model
print(clf.score(X_test, y_test))

# making predications using the machine learning model
# use a trained model to make predication
# use predict function
# use predict_proba

print(X_test)
print(clf.predict(X_test))
Y_pred = clf.predict(X_test)
print(np.mean(Y_pred == y_test))

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, Y_pred))

# predict Vs predict_proba()

#  make predication using predict_proba -> predict proba returns probabilities of classification
print(X_test[:5], y_test[:5])
print(clf.predict_proba(X_test[:5]))
print(heart_disease['target'].value_counts())

# making predications of our regression model

from sklearn.ensemble import RandomForestRegressor

np.random.seed(42)

# create the data
X = boston_df.drop('target', axis=1)
y = boston_df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestRegressor().fit(X_train, y_train)
# make predications
Y_preds = model.predict(X_test)
print(Y_preds[:10])

# Compare the predictions to the truth
from sklearn.metrics import mean_absolute_error

print(mean_absolute_error(y_test, Y_preds))

# Evaluating the model
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)
X = heart_disease.drop('target', axis=1)
y = heart_disease['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# 1: Estimator Score method

print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))

# Evaluating model using scoring parameter

from sklearn.model_selection import cross_val_score

np.random.seed(42)
X = heart_disease.drop('target', axis=1)
y = heart_disease['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# 1: Estimator Score method

print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))
print(cross_val_score(clf, X, y, cv=5))

