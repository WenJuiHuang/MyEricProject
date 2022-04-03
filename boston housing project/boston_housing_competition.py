"""
File: boston_housing_competition.py
Name: 
--------------------------------
This file demonstrates how to analyze boston
housing dataset. Students will upload their 
results to kaggle.com and compete with people
in class!

You are allowed to use pandas, sklearn, or build the
model from scratch! Go data scientist!
"""
import matplotlib.pyplot as plt
import math
import random
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import metrics
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import decomposition
from sklearn import tree
from sklearn import ensemble
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from time import time
from scipy.stats import randint as sp_randint


def main():
    data = pd.read_csv('boston_housing/train.csv')
    data.pop('ID')

    # data['lstat'] = np.log(data['lstat'])
    # data['age'] = np.power(data['age'], 2)
    # data['medv'] = np.log1p(data['medv'])

    x = data.drop(columns=['medv', 'rad'], axis=1)
    y = data['medv']


    # data split
    x_train, x_valid, y_train, y_valid = model_selection.train_test_split(x, y, test_size=0.5, random_state=0)

    # training with diff model

    # # linear regression
    # model = linear_model.LinearRegression()
    # training_acc(model, x_train, y_train)
    # valid_acc(model, x_valid, y_valid)

    # # linear regression with data normalization
    # normalizer = preprocessing.MinMaxScaler()
    # x_train = normalizer.fit_transform(x_train)
    # x_valid = normalizer.transform(x_valid)
    # model = linear_model.LinearRegression()
    # training_acc(model, x_train, y_train)
    # valid_acc(model, x_valid, y_valid)

    # # Polynomial
    # poly_phi_extractor = preprocessing.PolynomialFeatures(degree=2)
    # x_train = poly_phi_extractor.fit_transform(x_train)
    # x_valid = poly_phi_extractor.transform(x_valid)
    # model = linear_model.LinearRegression()
    # training_acc(model, x_train, y_train)
    # valid_acc(model, x_valid, y_valid)

    # # Polynomial with data normalization
    # normalizer = preprocessing.MinMaxScaler()
    # x_train = normalizer.fit_transform(x_train)
    # x_valid = normalizer.transform(x_valid)
    # poly_phi_extractor = preprocessing.PolynomialFeatures(degree=2)
    # x_train = poly_phi_extractor.fit_transform(x_train)
    # x_valid = poly_phi_extractor.transform(x_valid)
    # model = linear_model.LinearRegression()
    # training_acc(model, x_train, y_train)
    # valid_acc(model, x_valid, y_valid)

    # # Polynomial with data standardization
    # standardizer = preprocessing.StandardScaler()
    # x_train = standardizer.fit_transform(x_train)
    # x_valid = standardizer.transform(x_valid)
    # poly_phi_extractor = preprocessing.PolynomialFeatures(degree=2)
    # x_train = poly_phi_extractor.fit_transform(x_train)
    # x_valid = poly_phi_extractor.transform(x_valid)
    # model = linear_model.LinearRegression()
    # training_acc(model, x_train, y_train)
    # valid_acc(model, x_valid, y_valid)

    # # pca degree1
    # standardizer = preprocessing.StandardScaler()
    # x_train = standardizer.fit_transform(x_train)
    # x_valid = standardizer.transform(x_valid)
    # pca = decomposition.PCA(n_components=10)
    # x_train = pca.fit_transform(x_train)
    # x_valid = pca.transform(x_valid)
    # model = linear_model.LinearRegression()
    # training_acc(model, x_train, y_train)
    # valid_acc(model, x_valid, y_valid)

    # # pca degree2
    # standardizer = preprocessing.StandardScaler()
    # x_train = standardizer.fit_transform(x_train)
    # x_valid = standardizer.transform(x_valid)
    # pca = decomposition.PCA(n_components=7)
    # x_train = pca.fit_transform(x_train)
    # x_valid = pca.transform(x_valid)
    # poly_fea_extractor = preprocessing.PolynomialFeatures(degree=2)
    # x_train = poly_fea_extractor.fit_transform(x_train)
    # x_valid = poly_fea_extractor.transform(x_valid)
    # model = linear_model.LinearRegression()
    # training_acc(model, x_train, y_train)
    # valid_acc(model, x_valid, y_valid)

    # decision tree
    # model = tree.DecisionTreeRegressor(random_state=0, max_depth=6, max_leaf_nodes=25)
    # training_acc(model, x_train, y_train)
    # valid_acc(model, x_valid, y_valid)

    # randomforest
    model = ensemble.RandomForestRegressor(random_state=17, max_depth=9, min_samples_split=4)
    training_acc(model, x_train, y_train)
    valid_acc(model, x_valid, y_valid)

    # coef = pd.Series(model.feature_importances_, x.columns).sort_values(ascending=False)
    # coef.plot(kind='bar', title='Feature Importance')


    # # xgboost
    # model = xgb.XGBRegressor(seed=0, max_depth=6, min_child_weight=4)
    # training_acc(model, x_train, y_train)
    # valid_acc(model, x_valid, y_valid)

    # coef = pd.Series(model.coef_, x.columns).sort_values()
    # coef.plot(kind='bar',title='Model Coefficients')
    # plt.show()

    # predict
    data_test = pd.read_csv('boston_housing/test.csv')
    x_test = data_test.drop(columns=['ID','rad'], axis=1)
    pred = model.predict(x_test)
    # pred = np.exp(pred) - 1


    out_file(pred, 'boston_housing_competition.csv', data_test.ID)


def out_file(predictions, filename, test_id):
    """
	: param predictions: numpy.array, a list-like data structure that stores 0's and 1's
	: param filename: str, the filename you would like to write the results to
	: param test_id: numpy.array, IDs
	"""

    print('\n===============================================')
    print(f'Writing predictions to --> {filename}')
    with open(filename, 'w') as out:
        out.write('ID,medv\n')
        # start_id = 1
        for i in range(len(predictions)):
            out.write(str(test_id[i]) + ',' + str(predictions[i]) + '\n')
            # start_id += 1
    print('===============================================')


def training_acc(model, x_train, y_train):
    classifier = model.fit(x_train, y_train)
    acc_train = classifier.score(x_train, y_train)
    print('Training acc: ' + str(acc_train))


def valid_acc(model, x_valid, y_valid):
    pred = model.predict(x_valid)
    mse = metrics.mean_squared_error(y_valid, pred)
    print('Valid RMS Error: ' + str(mse**0.5))



if __name__ == '__main__':
    main()
