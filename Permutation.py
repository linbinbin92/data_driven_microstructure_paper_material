#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 15:08:28 2019

@author: binbin
"""

## import some libriaries ##
import pandas as pd
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing

from sklearn.neural_network import MLPRegressor
from sklearn import ensemble

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate, cross_val_predict

import matplotlib.pyplot as plt
from matplotlib import cm as cm
import plotly.express as px

def draw_train_test_kde(feature_name,train_df,test_df):
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.distplot(train_df[feature_name], color=sns.color_palette("coolwarm",5)[0], label='Train')
    sns.distplot(test_df[feature_name], color=sns.color_palette("coolwarm",5)[4], label='Test')
    ax.set_title('Comparison of the ' + feature_name + ' distribution', size=20);


####################Data Preparation ##############################

df=pd.read_pickle('df.pkl')

## drop some features that were addtionally included and but not used ##
df=df.drop(['SVF Mean Y'],axis=1)
df=df.drop(['SVF Mean X'],axis=1)
df=df.drop(['SVF Mean Z'],axis=1)
df=df.drop(['SVF STD Y'],axis=1)
df=df.drop(['SVF STD X'],axis=1)
df=df.drop(['SVF STD Z'],axis=1)
df=df.drop(['Realsurface'],axis=1)
df=df.drop(['Thoughness'],axis=1)


#### some specific scalling parameters##

scale_para=30e7
FN_len=0.4
area=0.05*FN_len
para=scale_para/area

df['Strain to failure']=df['Strain to failure']*100/FN_len
df['Maximal Stress']=df['Maximal Stress']*para/1e6
df['Initial Effective Stiffness']=df['Initial Effective Stiffness']*para/(100/FN_len)
dataset2=df.copy()

##resemble and naming for the correlation after analysis of the heat plot ##

dataset2=dataset2.rename(columns={'Orientation STD': 'F-O STD', 'Length STD':'F-L STD',"Diameter STD": "F-D STD",
                        "Cont_Area_X_Mean":"C-ASD_X Mean",
                        "Cont_Area_X_STD":"C-ASD_X STD",
                        "Cont_Area_Y_Mean":"C-ASD_Y Mean",
                        "Cont_Area_Y_STD":"C-ASD_Y STD",
                        "Cont_Area_Z_Mean":"C-ASD_Z Mean",
                        "Cont_Area_Z_STD":"C-ASD_Z STD",
                        "Cont_Area_Normal_X_Mean":"C-ANO_X Mean",
                        "Cont_Area_Normal_X_STD":"C-ANO_X STD",
                        "Cont_Area_Normal_Y_Mean":"C-ANO_Y Mean",
                        "Cont_Area_Normal_Y_STD":"C-ANO_Y STD",
                        "Cont_Area_Normal_Z_Mean":"C-ANO_Z Mean",
                        "Cont_Area_Normal_Z_STD":"C-ANO_Z STD",
                        "Cont_Area_Size_Mean":"C-AS Mean",
                        "Cont_Area_Size_STD":"C-AS STD"})

dataset3=dataset2.copy()

## Drop ones with high correlation ##
dataset3=dataset3.drop(['C-ASD_Z Mean'],axis=1)
dataset3=dataset3.drop(['C-ANO_X STD'],axis=1)
dataset3=dataset3.drop(['C-ANO_Y STD'],axis=1)
dataset3=dataset3.drop(['C-AS STD'],axis=1)
dataset3=dataset3.drop(['C-ANO_Z STD'],axis=1)
dataset3=dataset3.drop(['C-ANO_Z Mean'],axis=1)

## setting the features for plotting purposes##
A=list(range(0,20))
targets_index=18
A.pop(targets_index)

X0=dataset2.iloc[:,0:17]
Y0=dataset2.iloc[:,17:20]

a = 13 ## choice of output to evaluate
X=dataset3.iloc[:,0:11]
y=dataset3.iloc[:,a]

###################### potting the correlation of the data #####################

##before selecting##
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
corr = spearmanr(X0).correlation
corr_linkage = hierarchy.ward(corr)
dendro = hierarchy.dendrogram(corr_linkage, labels=X0.columns, ax=ax1,
                              leaf_rotation=90)
dendro_idx = np.arange(0, len(dendro['ivl']))

clo=ax2.imshow(corr[dendro['leaves'], :][:, dendro['leaves']])

ax2.set_xticks(dendro_idx)
ax2.set_yticks(dendro_idx)
ax2.set_xticklabels(dendro['ivl'], rotation='vertical')
ax2.set_yticklabels(dendro['ivl'])
fig.tight_layout()
fig.colorbar(clo)
plt.savefig("Feature_Selection_before.svg",format="svg")
plt.show()

##after selection##

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
corr = spearmanr(X).correlation
corr_linkage = hierarchy.ward(corr)
dendro = hierarchy.dendrogram(corr_linkage, labels=X.columns, ax=ax1,
                              leaf_rotation=90)
dendro_idx = np.arange(0, len(dendro['ivl']))
clo=ax2.imshow(corr[dendro['leaves'], :][:, dendro['leaves']])
clo1=corr[dendro['leaves'], :][:, dendro['leaves']]
ax2.set_xticks(dendro_idx)
ax2.set_yticks(dendro_idx)
ax2.set_xticklabels(dendro['ivl'], rotation='vertical')
ax2.set_yticklabels(dendro['ivl'])
fig.tight_layout()
fig.colorbar(clo)
plt.savefig("Feature_Selection_after.svg",format="svg")
plt.show()

################# Permutation model training ####################

scalerSS= StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15,
                                                    random_state=10)

try:
    y_train.shape[1]
except IndexError:
    y_train=y_train.values.reshape(-1,1)
    y_test=y_test.values.reshape(-1,1)

X_train=scalerSS.fit_transform(X_train)
X_test = scalerSS.transform(X_test)

y_train=scalerSS.fit_transform(y_train)
y_test=scalerSS.transform(y_test)


mlp= ensemble.GradientBoostingRegressor()

MLP=mlp.fit(X_train, y_train)
y_pred=mlp.predict(X_test)
mseMLP = mean_squared_error(y_test, y_pred)
print("MSE MLP: %.4f" % mseMLP)
print("Train R2 score MLP: {:.2f}".format(mlp.score(X_train, y_train)))
print("Test R2 score MLP: {:.2f}".format(mlp.score(X_test, y_test)))
score_MLP= cross_val_score(MLP, X_test, y_test,scoring='r2')
print("Test CV-error MLP: {:.2f}".format(score_MLP.mean()))


if a == 11:
    result_strain = permutation_importance(mlp, X_test, y_test, n_repeats=100,
                                    random_state=42, n_jobs=10)
    print('strain is selected')
elif a == 12:

    result_stress = permutation_importance(mlp, X_test, y_test, n_repeats=100,
                                    random_state=42, n_jobs=10)
    print('stress is selected')

elif a == 13:
    result_stiff = permutation_importance(mlp, X_test, y_test, n_repeats=100,
                                    random_state=42, n_jobs=10)
    print('stiffness is selected')


## strain##

result_PI_strain = (100.0*(result_strain.importances/result_strain.importances.max()))
PI_to_Strain=result_strain.importances_mean
df_RI_strain=pd.DataFrame()
df_RI_strain['IP_mean'] =  abs(100.0*result_strain.importances_mean/result_strain.importances.max())
df_RI_strain['IP_std'] = result_strain.importances_std
df_RI_strain['Features'] = X.columns
df_RI_strain['Targets'] = 'Strain to failure'


## stress ##

result_PI_stress = 100.0*(result_stress.importances/result_stress.importances.max())
PI_to_Stress=result_stress.importances_mean

df_RI_stress=pd.DataFrame()
df_RI_stress['IP_mean'] =  abs(100.0*result_stress.importances_mean/result_stress.importances.max())
df_RI_stress['IP_std'] = result_stress.importances_std
df_RI_stress['Features'] = X.columns
df_RI_stress['Targets'] = 'Maximal stress'


## stiffness ##

result_PI_stiff = 100.0*(result_stiff.importances/result_stiff.importances.max())
PI_to_Stiffness=result_stiff.importances_mean

df_RI_stiff=pd.DataFrame()
df_RI_stiff['IP_mean'] = abs(100.0*result_stiff.importances_mean/result_stiff.importances.max())
df_RI_stiff['IP_std'] = result_stiff.importances_std
df_RI_stiff['Features'] = X.columns
df_RI_stiff['Targets'] = 'Effective stiffness'

## grouping ##

df_boxplot = pd.DataFrame()
df_boxplot=df_boxplot.append(df_RI_stiff)
df_boxplot=df_boxplot.append(df_RI_strain)
df_boxplot=df_boxplot.append(df_RI_stress)

#################Sensitivity permutation ####################################

import matplotlib.style
import matplotlib as mpl
mpl.style.use('classic')

def grouped_barplot(df, cat,subcat, val , err):
    u = df[cat].unique()
    x = np.arange(len(u))
    subx = df[subcat].unique()
    offsets = (np.arange(len(subx))-np.arange(len(subx)).mean())/(len(subx)+1.)
    width= np.diff(offsets).mean()
    fig = plt.figure()
    ax = fig.add_subplot()

    ax.tick_params(axis='both', labelsize=13)

    ax.yaxis.label.set_size(15)
#    ax.set_facecolor('white')
    ax.xaxis.label.set_size(15)
    fig.patch.set_facecolor('xkcd:white')
    for i,gr in enumerate(subx):
        dfg = df[df[subcat] == gr]
        ax.bar(x+offsets[i], dfg[val].values, width=width,
                label="{}".format(gr))

    plt.xlabel('Features')
    plt.ylabel('Relative Importance based on Permutation')
    plt.xticks(x, u, rotation=45)
    plt.legend()
    plt.style.use('ggplot')
    plt.tight_layout()
    plt.savefig("Sensitivity_permutation.svg",format="svg")

    plt.show()


cat = "Features"
subcat = "Targets"
val = "IP_mean"
err = "IP_std"
plt.savefig("Sensitivity_permutation.svg",format="svg")

grouped_barplot(df_boxplot, cat, subcat, val,err )


#################Sensitivity PCC ###############################

def grouped_barplot2(df, cat,subcat, val , err):
    u = df[cat].unique()
    x = np.arange(len(u))
    subx=[ 'Effective stiffness','Strain to failure','Maximal Stress']
    print(type(subx))
    offsets = (np.arange(len(subx))-np.arange(len(subx)).mean())/(len(subx)+1.)
    width= np.diff(offsets).mean()
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.tick_params(axis='both', labelsize=13)
    ax.yaxis.label.set_size(15)
    ax.xaxis.label.set_size(15)
    fig.patch.set_facecolor('xkcd:white')
    for i,gr in enumerate(subx):
        dfg = df[df[subcat] == gr]
        ax.bar(x+offsets[i], dfg[val].values, width=width,
                label="{}".format(gr))

    plt.xlabel('Features')
    plt.ylabel('Relative Importance based on PCC')
    plt.xticks(x, u, rotation=45)
    plt.legend()
    plt.style.use('ggplot')
    plt.tight_layout()
    plt.savefig("Sensitivity_PCC.svg",format="svg")

df_corr=dataset3.corr()
df_corr
df_corr3=df_corr.iloc[-3:]   ##### targets
df_PCC = df_corr3.T
df_PCC['Features'] = df_PCC.index

df_PCC_strain= df_PCC.copy().drop(['Maximal Stress'],axis=1)
df_PCC_strain=df_PCC_strain.drop(['Initial Effective Stiffness'],axis=1)
df_PCC_strain['Targets'] = 'Strain to failure'
maxStrain=df_PCC_strain['Strain to failure'].max()
df_PCC_strain['Strain to failure']=abs(100*df_PCC_strain['Strain to failure']/maxStrain)
df_PCC_strain=df_PCC_strain.rename(columns={'Strain to failure': 'Value'})
df_PCC_strain=df_PCC_strain.iloc[:-3]   ##### targets

df_PCC_stress= df_PCC.copy().drop(['Strain to failure'],axis=1)
df_PCC_stress=df_PCC_stress.drop(['Initial Effective Stiffness'],axis=1)
df_PCC_stress['Targets'] = 'Maximal Stress'
maxStress=df_PCC_stress['Maximal Stress'].max()
df_PCC_stress['Maximal Stress']=abs(100*df_PCC_stress['Maximal Stress']/maxStress)
df_PCC_stress=df_PCC_stress.rename(columns={'Maximal Stress': 'Value'})
df_PCC_stress=df_PCC_stress.iloc[:-3]   ##### targets

df_PCC_stiffness= df_PCC.copy().drop(['Strain to failure'],axis=1)
df_PCC_stiffness=df_PCC_stiffness.drop(['Maximal Stress'],axis=1)
df_PCC_stiffness['Targets'] = 'Effective stiffness'
maxStiffness=df_PCC_stiffness['Initial Effective Stiffness'].max()
df_PCC_stiffness['Initial Effective Stiffness']=abs(100*df_PCC_stiffness['Initial Effective Stiffness']/maxStiffness)

df_PCC_stiffness=df_PCC_stiffness.rename(columns={'Initial Effective Stiffness': 'Value'})
df_PCC_stiffness=df_PCC_stiffness.iloc[:-3]   ##### targets

df_PPCC = pd.DataFrame()
df_PPCC=df_PPCC.append(df_PCC_strain,ignore_index = True)
df_PPCC=df_PPCC.append(df_PCC_stress,ignore_index = True)
df_PPCC=df_PPCC.append(df_PCC_stiffness,ignore_index = True)


cat = "Features"
subcat = "Targets"
val = "Value"
err = "IP_std"

grouped_barplot2(df_PPCC, cat, subcat, val, err )
