# introduction to sci-kit learn
'''
This file demonstrates some of the most useful functions of the beautiful scikit learn librabry

0: An end to end scikit learn workflow
1: Getting the data ready
2: Choose the right estimator/algorithm for our problem
3: Fit the model/algorithm and use it to make predictions on our data
4: evaluting the model
5: improve the model
6: Save and load a trained model
7: Putting together all

'''

import pandas as pd
import numpy as np

heart_disease = pd.read_csv('../../data/datasets_33180_43520_heart.csv')
# print((heart_disease))
X = heart_disease.drop('target', axis=1)
y = heart_disease['target']

print(X)
print(y)

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()
clf.get_params()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf.fit(X_train, y_train)
# make prediction
y_pred = clf.predict(X_test)
print(y_pred)

# Evaluate model

print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

# improve model

np.random.seed(42)
for i in range(10, 100, 10):
    print(f"Trying model with {i} estimators...")
    clf = RandomForestClassifier(n_estimators=i)
    clf.fit(X_train, y_train)
    print(f"Model accuracy on test set :{clf.score(X_test, y_test) * 100:.2f} %")
    print("")

# save the model

import pickle

pickle.dump(clf, open("Random_forect.pkl", "wb"))
loaded_model = pickle.load(open("Random_forect.pkl", "rb"))
print(loaded_model.score(X_test, y_test))
