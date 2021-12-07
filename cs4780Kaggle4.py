
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn import metrics
import pandas as pd
import pickle as pkl
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor


def score(preds, y_val):
    score = 0
    for i in range(len(preds)):
        score += (preds[i]-y_val[i])**2
    return score / len(preds)


def deleteFeature(x_train, y_train, x_val, y_val, x_test, bestScore):
    temp_x_train = x_train
    temp_y_train = y_train
    temp_x_val = x_val
    temp_x_test = x_test

    # for i in range(len(x_train[0])):
    # x_train[0] doesn't update, so if i = 23, we cant delete 22nd index
    # from a shorter array, so i jsut removed 1st
    features_drop = []
    i = len(x_train[0]) - 1
    print(i)
    while (i != 0):
        temp_x_train_2 = temp_x_train
        temp_y_train_2 = temp_y_train
        temp_x_val_2 = temp_x_val
        temp_x_test_2 = temp_x_test

        x_train_shape = []
        # print(i)
        # print(len(temp_x_train_2[0]))
        for f in range(len(temp_x_train_2)):
            temp = np.delete(temp_x_train_2[f], len(temp_x_train[0]) - 1 - i)
            x_train_shape.append(temp)
        temp_x_train_2 = x_train_shape
        # print(len(temp_x_train_2[0]))

        x_val_shape = []
        for j in range(len(temp_x_val_2)):
            temp = np.delete(temp_x_val_2[j], len(temp_x_val[0]) - 1 - i)
            x_val_shape.append(temp)
        temp_x_val_2 = x_val_shape

        x_test_shape = []
        for z in range(len(temp_x_test_2)):
            temp = np.delete(temp_x_test_2[z], len(temp_x_test[0]) - 1 - i)
            x_test_shape.append(temp)
        temp_x_test_2 = x_test_shape

        LR_new = XGBRegressor()
        LR_new.fit(temp_x_train_2, temp_y_train_2)
        y_pred = LR_new.predict(temp_x_val_2)
        currentScore = score(y_pred, y_val)
       # print(currentScore)
        if currentScore < bestScore:
            bestScore = currentScore
            i = len(temp_x_test[0]) - 1
            features_drop.append(i)
        print(bestScore)
        i = i - 1

    return features_drop


# data = pd.read_pickle("covid_dataset.pkl")

# data = pd.DataFrame([data])
# # replace NaNs with mean
# data.fillna(0)
# data.replace([np.inf, -np.inf], 0)

# # data = data.astype(object)

# could just do numpy arrays?
file = open('covid_dataset.pkl', 'rb')
checkpoint = pkl.load(file)
file.close()

X_train, y_train, X_val, y_val, X_test = checkpoint["X_train"], checkpoint[
    "y_train_log_pos_cases"], checkpoint["X_val"], checkpoint["y_val_log_pos_cases"], checkpoint["X_test"]
x_train = []

# X_train is made of countries, each country has its own list of values wtih features

'''
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

'''
feature_means_train = []
for i in range(23):
    i_sum = 0
    counter = 0
    for x in range(len(X_train)):
        if not X_train[x][i] == None:
            i_sum += X_train[x][i]
            counter += 1
    feature_means_train.append(i_sum / counter)

feature_means_val = []
for i in range(23):
    i_sum = 0
    counter = 0
    current_largest = X_val[0][i]
    for x in range(len(X_val)):
        if not X_val[x][i] == None:
            i_sum += X_val[x][i]
            counter += 1
    feature_means_val.append(i_sum / counter)


feature_means_test = []
for i in range(23):
    i_sum = 0
    counter = 0
    for x in range(len(X_test)):
        if not X_test[x][i] == None:
            i_sum += X_test[x][i]
            counter += 1
    feature_means_test.append(i_sum / counter)

x_train = []
for x in (X_train):
    for i in range(23):
        if x[i] == None:
            x[i] = feature_means_train[i]
    x_train.append(x)

x_val = []
for x in X_val:
    for i in range(23):
        if x[i] == None:
            x[i] = feature_means_val[i]
    x_val.append(x)

x_test = []
for x in X_test:
    for i in range(23):
        if x[i] == None:
            x[i] = feature_means_test[i]
    x_test.append(x)

# dummy numbers i created
# x_train = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
# y_train = np.array([13, 14, 15])
# x_val = np.array([[17, 18, 19, 20], [21, 22, 23, 24]])
# y_val = np.array([1, 2])
# x_test = np.array([[25,26, 27, 28], [30, 31, 32, 33]])


LR = XGBRegressor()
LR.fit(x_train, y_train)
y_pred = LR.predict(x_val)

test_pred = y_pred
initial_score = score(y_pred, y_val)

x_train_shape = []
features_drop = []
features_drop = deleteFeature(
    x_train, y_train, x_val, y_val, x_test, initial_score)
print(x_train[0])
for i in range(len(features_drop)):
    x_train_shape = []
    x_test_shape = []
    x_val_shape = []
    for j in range(len(x_train)):
        temp = np.delete(x_train[j], features_drop[i])
        x_train_shape.append(temp)
    x_train = x_train_shape
    for j in range(len(x_val)):
        temp = np.delete(x_val[j], features_drop[i])
        x_val_shape.append(temp)
    x_val = x_val_shape
    for j in range(len(x_test)):
        temp = np.delete(x_test[j], features_drop[i])
        x_test_shape.append(temp)
    x_test = x_test_shape
print(len(x_train[0]))


'''
clf = Ridge(alpha=1.0)
clf.fit(X, y)
'''
LR_2 = XGBRegressor()
LR_2.fit(x_train, y_train)
# LR_2.fit(x_train_new, y_train_new)

test_pred = LR_2.predict(x_test)


# go through the predictions and subtract if greater than mean
mean = np.mean(test_pred)
num = len(test_pred)
for i in range(num):
    if test_pred[i] > mean * 1.5:
        test_pred[i] -= mean * .75
    if test_pred[i] < mean * .5:
        test_pred[i] += mean*.25
for i in range(len(test_pred)):
    print(i)
    print(test_pred[i])


pd.DataFrame(test_pred).to_csv("predictions.csv",
                               header=["cases"], index_label="id")


# score function
# drop a feature and use model with x val
# compare with y_val
