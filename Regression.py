#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 15:08:28 2019

@author: binbin
"""

## import some libriaries ##

import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import RobustScaler

from distutils.version import LooseVersion
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.svm import SVR
from sklearn import neighbors
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import BayesianRidge, LinearRegression

from sklearn.metrics import median_absolute_error, r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate, cross_val_predict

from sklearn import ensemble
from sklearn.pipeline import make_pipeline
from scipy.stats.stats import pearsonr


import matplotlib.ticker as ticker
def plot_regression_results(ax, y_true, y_pred, title, scores):
    """Scatter plot of the predicted vs true targets."""
    ax.plot([y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()],
            '--k', linewidth=2)
    ax.scatter(y_true, y_pred,c='b',alpha =0.3)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    ax.set_xlim([y_true.min(), y_true.max()])
    ax.set_ylim([y_true.min(), y_true.max()])
    ax.set_xlabel('FEM [Pa]')
    ax.set_ylabel('ML-Predicted [Pa]')
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                          edgecolor='none', linewidth=0)
    ax.legend([extra], [scores], loc='upper left')
    ax.set_title(title)

    title = title
    SMALL_SIZE = 8
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 15
    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


########################### Data preparation ##############
## read data##
df = pd.read_pickle('df.pkl')

## some specific scalling parameters##
scale_para=30e7 # in Pa
area=0.05*0.4 # in mm^2
para=scale_para/area
FN_len=0.4 # mm

## define output##
df['Strain to failure']=df['Strain to failure']*100/FN_len
df['Maximal Stress'] = df['Maximal Stress']*para/1e6
df['Initial Effective Stiffness'] = df['Initial Effective Stiffness']*para/(100/FN_len)

## drop some outliners##
df = df.dropna()
df=df.drop(['41_1.2e-05_4.4e-06'],axis=0)
df=df.drop(['45_2.7e-05_2.2e-06'],axis=0)
df=df.drop(['11_1e-05_4.7e-06'],axis=0)
df=df.drop(['10_2.9e-05_5e-06'],axis=0)

## prepare train and test dataset##
dataset = df.copy()
dataset=dataset.reset_index()
dataset=dataset.drop(['index'], axis=1)
dataset=dataset.astype('float64')

train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

## define output##
# note Maximal Stress can be changed to strain or stiffness
train_labels = train_dataset.pop('Maximal Stress')
test_labels = test_dataset.pop('Maximal Stress')


##Scalling the dataset##
# any other scaler can be used, see imported scaler above#

ss = StandardScaler()
# fit_transform()先拟合数据，再标准化
normed_train_data = ss.fit_transform(train_dataset)
# transform()数据标准化
normed_test_data = ss.transform(test_dataset)

X_train = normed_train_data
y_train = train_labels
X_test  = normed_test_data
y_test  = test_labels

########################Train and eval ######################

estimators = [
    ('LinearRegression',LinearRegression()),
    ('Lasso', LassoCV()),
    ('KNN', neighbors.KNeighborsRegressor()),
    ('GradientBosting', ensemble.HistGradientBoostingRegressor())]
#stacking_regressor = SVR()
#    ('Random Forest', RandomForestRegressor(random_state=42)),

fig, axs = plt.subplots(1, 4, figsize=(20, 7))
axs = np.ravel(axs)

for ax, (name, est) in zip(axs, estimators):
    score = cross_validate(est, X_train, y_train,
                           scoring=['r2', 'neg_mean_absolute_error'],
                           n_jobs=-1, verbose=0)

    y_pred = cross_val_predict(est, X_test, y_test, n_jobs=-1, verbose=0)
    plot_regression_results(
        ax, y_test, y_pred, name,
        (r'$R^2={:.2f} \pm {:.2f}$' + "\n" +r'$MAE={:.2e} \pm {:.2e}$' ).format(
                np.mean(score['test_r2']),
                np.std(score['test_r2']),
                -np.mean(score['test_neg_mean_absolute_error']),
                np.std(score['test_neg_mean_absolute_error'])))

plt.suptitle('Cross-validated results on Maximal Stress')
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()
