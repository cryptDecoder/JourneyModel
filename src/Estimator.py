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

