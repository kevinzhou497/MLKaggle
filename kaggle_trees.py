
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn import metrics
from sklearn import tree
import seaborn as sns
from sklearn import preprocessing
from sklearn import utils

import matplotlib.pyplot as plt

import pandas as pd
import pickle as pkl
import numpy as np
import xgboost
from xgboost import XGBRegressor

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingClassifier

# potential improvements
# 1. change feature selection process
# 2. tune parameters of XGB
# 3. changing training and validation


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

        tree = DecisionTreeClassifier()
        LR_new = BaggingClassifier(tree, n_estimators=3, max_samples=.2,
                        random_state=5)

        LR_new.fit(temp_x_train_2, temp_y_train_2)
        y_pred = LR_new.predict(temp_x_val_2)


        # clf=RandomForestClassifier(n_estimators=100)
        # lab_enc = preprocessing.LabelEncoder()
        # y_train = lab_enc.fit_transform(y_train)
        # clf.fit(temp_x_train_2,temp_y_train_2)
        # test_pred=clf.predict(temp_x_val_2)

        currentScore = score(y_pred, y_val)
        #print(currentScore)
        if currentScore < bestScore:
            bestScore = currentScore
            temp_x_train = temp_x_train_2
            temp_y_train = temp_y_train_2
            temp_x_val = temp_x_val_2
            temp_x_test = temp_x_test_2
            i = len(temp_x_test[0]) - 1
        print(bestScore)
        i = i - 1

    return temp_x_train, temp_y_train, temp_x_test


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


# replaces nones with mean of a given feature for each country
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
china_index = 0
for x in (X_train):
    china_index += 1
    for i in range(23):
        if x[i] == None:
            x[i] = feature_means_train[i]
        # if x[i] > 1000000000:
        #     print(china_index) #said china index was 10
    if china_index != 10:   
      x_train.append(x)
print(len(x_train))


temp = []
for y_l in range(len(y_train)):
    if y_l != 10:
      temp.append(y_train[y_l])
y_train = temp
print(len(y_train))


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

tree = DecisionTreeClassifier()
bag = BaggingClassifier(tree, n_estimators=3, max_samples=.2,
                        random_state=5)
lab_enc = preprocessing.LabelEncoder()
y_train = lab_enc.fit_transform(y_train)
bag.fit(x_train, y_train)

y_pred=bag.predict(x_val)


# clf=RandomForestClassifier(n_estimators=100)
# lab_enc = preprocessing.LabelEncoder()
# y_train = lab_enc.fit_transform(y_train)
# clf.fit(x_train,y_train)
# y_pred=clf.predict(x_test)



test_pred = y_pred
initial_score = score(y_pred, y_val)

x_train_new, y_train_new, x_test_new = deleteFeature(
    x_train, y_train, x_val, y_val, x_test, initial_score)


'''
clf = Ridge(alpha=1.0)
clf.fit(X, y)
'''

tree = DecisionTreeClassifier()
bag = BaggingClassifier(tree, n_estimators=3, max_samples=.2,
                        random_state=5)

bag.fit(x_train_new, y_train_new)

# #clf_2 = tree.DecisionTreeRegressor()
# #clf_2.fit(x_train_new, y_train_new)
# # LR_2.fit(x_train_new, y_train_new)

test_pred = bag.predict(x_test_new)

# clf=RandomForestClassifier(n_estimators=100)
# lab_enc = preprocessing.LabelEncoder()
# y_train = lab_enc.fit_transform(y_train)
# clf.fit(x_train,y_train)
# test_pred=clf.predict(x_test)


# go through the predictions and subtract if greater than mean
mean = np.mean(test_pred)
num = len(test_pred)
for i in range(num):
    if test_pred[i] > mean * 1.3:
        test_pred[i] -= mean * .4
    if test_pred[i] < mean * .5:
        test_pred[i] += mean*.25


pd.DataFrame(test_pred).to_csv("predictions.csv",
                               header=["cases"], index_label="id")


# score function
# drop a feature and use model with x val
# compare with y_val
