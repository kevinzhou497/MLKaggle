from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import pandas as pd
import pickle as pkl
import numpy as np

file = open('covid_dataset.pkl', 'rb')
checkpoint = pkl.load(file)
file.close()

#data = pd.read_pickle("covid_dataset.pkl")

# data = pd.DataFrame([data])
# # replace NaNs with mean
# data.fillna(0)
# data.replace([np.inf, -np.inf], 0)

# # data = data.astype(object)

# could just do numpy arrays?


X_train, y_train, X_val, y_val, X_test = checkpoint["X_train"], checkpoint[
    "y_train_log_pos_cases"], checkpoint["X_val"], checkpoint["y_val_log_pos_cases"], checkpoint["X_test"]
x_train = []


for x in X_train:
    x[x == None] = 0
    mean = np.mean(x)
    x[x == 0] = 0
    x_train.append(x)

x_val = []
for x in X_val:
    x[x == None] = 0
    mean = np.mean(x)
    x[x == 0] = 0
    x_val.append(x)

x_test = []
for x in X_test:
    x[x == None] = 0
    mean = np.mean(x)
    x[x == 0] = 0
    x_test.append(x)

LR = LinearRegression()
LR.fit(x_train, y_train)
y_pred = LR.predict(x_test)
print(y_pred)

test_pred = y_pred
pd.DataFrame(test_pred).to_csv("predictions.csv",
                               header=["cases"], index_label="id")
