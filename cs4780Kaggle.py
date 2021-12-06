
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import pandas as pd
import pickle as pkl
import numpy as np


def score(preds, y_val):
    score = 0
    for i in range(len(preds)):
        score += (preds[i]-y_val[i])**2
    return score / len(preds)


def deleteFeature(x_train, y_train, x_val, y_val, bestScore):
    temp_x_train = x_train
    temp_y_train = y_train
    for i in range(len(x_train[0])):
        temp_x_train_2 = temp_x_train
        temp_y_train_2 = temp_y_train
        for f in temp_x_train_2:
            np.delete(f, i)
        # for fy in temp_y_train_2:
           # np.delete(fy, i)

        LR_new = LinearRegression()
        LR_new.fit(temp_x_train, temp_y_train)
        y_pred = LR.predict(x_val)
        currentScore = score(y_pred, y_val)
        if currentScore < bestScore:
            bestScore = currentScore
            temp_x_train = temp_x_train_2
            temp_y_train = temp_y_train_2
    return temp_x_train, temp_y_train


file = open('covid_dataset.pkl', 'rb')
checkpoint = pkl.load(file)
file.close()

# data = pd.read_pickle("covid_dataset.pkl")

# data = pd.DataFrame([data])
# # replace NaNs with mean
# data.fillna(0)
# data.replace([np.inf, -np.inf], 0)

# # data = data.astype(object)

# could just do numpy arrays?


X_train, y_train, X_val, y_val, X_test = checkpoint["X_train"], checkpoint[
    "y_train_log_pos_cases"], checkpoint["X_val"], checkpoint["y_val_log_pos_cases"], checkpoint["X_test"]
x_train = []

# X_train is made of countries, each country has its own list of values wtih features

for x in X_train:
    x[x == None] = 0
    mean = np.mean(x)
    x[x == 0] = mean
    x_train.append(x)

x_val = []
for x in X_val:
    x[x == None] = 0
    mean = np.mean(x)
    x[x == 0] = mean
    x_val.append(x)

x_test = []
for x in X_test:
    x[x == None] = 0
    mean = np.mean(x)
    x[x == 0] = mean
    x_test.append(x)

LR = LinearRegression()
LR.fit(x_train, y_train)
y_pred = LR.predict(x_val)

test_pred = y_pred
initial_score = score(y_pred, y_val)
print(initial_score)
x_train, y_train = deleteFeature(x_train, y_train, x_val, y_val, initial_score)
LR = LinearRegression()
LR.fit(x_train, y_train)
test_pred = LR.predict(x_test)

pd.DataFrame(test_pred).to_csv("predictions.csv",
                               header=["cases"], index_label="id")


# score function
# drop a feature and use model with x val
# compare with y_val
