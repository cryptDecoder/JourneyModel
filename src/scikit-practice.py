import numpy as np
import pandas as pd

# get data ready

heart_disease = pd.read_csv("../data/datasets_33180_43520_heart.csv")
print(heart_disease.head())

# create x and y using data
X = heart_disease.drop('target', axis=1)
y = heart_disease['target']
print("Values of X :\n", X)
print("Values of y :\n", y)

# choose the right estimator/model for our problem
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()
print(clf.get_params())

# split the data into test and train

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# printing data
print("X train data \n", X_train)
print("y train data :\n", y_train)
print("X test data :\n", X_test)
print("Y test data :\n", y_test)

# fit the model
clf.fit(X_train, y_train)

# make predication
y_pred = clf.predict(X_test)
print(y_pred)

# Evaluate Model
print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("Classification Report : \n", classification_report(y_test, y_pred))
print("Confussion Metrix :\n", confusion_matrix(y_test, y_pred))
print("Accuracy :\n", accuracy_score(y_test, y_pred))

# improve model
np.random.seed(42)
for i in range(10, 100, 10):
    print(f"Trying model with estimator {i}")
    clf = RandomForestClassifier().fit(X_train, y_train)
    print(f"Model accuracy on test set :{clf.score(X_test, y_test) * 100:2f}%")
    print("")

# save model
import pickle

pickle.dump(clf, open("Random_forect.pkl", "wb"))

# used saved model
loaded_model = pickle.load(open("Random_forect.pkl", "rb"))
print("loaded model score : ", loaded_model.score(X_test, y_test))
