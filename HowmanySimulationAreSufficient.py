#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: b.lin@mfm.tu-darmstadt.de
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
import random
from sklearn import preprocessing

def Mean_Val_Con(D_strain):
    Mean_Val=[]
    N=D_strain.size
    for i in range(N-1):
    #    print(D_strain[0:i+1])
        Mean=D_strain[0:i+1].mean()
        Mean_Val.append(Mean)
    Mean_Val=pd.DataFrame(Mean_Val)
    MM=Mean_Val.mean()
    Sx=Mean_Val.std()

    Z=2.576

    MMU=MM + -Z*Sx/np.sqrt(N)
    MMO=MM + Z*Sx/np.sqrt(N)

    plt.figure()
    plt.plot(range(N-1),Mean_Val)
    plt.axhline(y=MM.item(), xmin=0, xmax=N-1, linewidth=1, color = 'y',label='Mean value')
    plt.axhline(y=MMU.item(), xmin=0, xmax=N-1, linewidth=1, color = 'r',linestyle='dashed',label='1-percentile errorbound')
    plt.axhline(y=MMO.item(), xmin=0, xmax=N-1, linewidth=1, color = 'r',linestyle='dashed')
    plt.ylabel(str(D_strain.name))
    plt.xlabel('Number of Simulation')
    plt.title('Mean Value with Number of Simulation')
    plt.legend()

    return  plt , Mean_Val

def STD_Val_Con(D_strain):

    STD_Val=[]
    N=D_strain.size
    for i in range(N-1):
        if i==0:
            pass
        else:
            STD=D_strain[0:i+1].mean()
            STD_Val.append(STD)
    STD_Val=pd.DataFrame(STD_Val)
    MM=STD_Val.mean()
    Sx=STD_Val.std()
    Z=2.576

    MMU=MM + -Z*Sx/np.sqrt(N)
    MMO=MM + Z*Sx/np.sqrt(N)

    plt.figure()
    plt.plot(range(N-2),STD_Val)
    plt.axhline(y=MM.item(), xmin=0, xmax=N-1, linewidth=1, color = 'y')
    plt.axhline(y=MMU.item(), xmin=0, xmax=N-1, linewidth=1, color = 'r',linestyle='dashed')
    plt.axhline(y=MMO.item(), xmin=0, xmax=N-1, linewidth=1, color = 'r',linestyle='dashed')
    plt.ylabel(str(D_strain.name))
    plt.xlabel('Number of Simulation')
    plt.title('Mean Value of STD with Number of Simulation')
    return plt ,STD_Val


def KeyToDelete(np_file):
    KeyToDelete=[]
    for key in np_file.keys():
        if len(np_file[key]) != 28:
            KeyToDelete.append(key)
    for i in KeyToDelete:
        del np_file[i]
    return np_file

def Convert2Pandas(np_file):

    df = pd.DataFrame(np_file)
    df=df.transpose()
    return df

def Convert2Pandas2(np_file):

    df = pd.DataFrame(np_file)
    df=df.rename(columns={0:'Orientation STD', 1:'Length STD', 2:'Diameter STD', 3:'SVF Mean X', 4:'SVF STD X',
                                5:'SVF Mean Y', 6:'SVF STD Y', 7:'SVF Mean Z', 8:'SVF STD Z', 9:'Realsurface', 10:'Cont_Area_X_Mean', 11:'Cont_Area_X_STD',
                                12:'Cont_Area_Y_Mean', 13:'Cont_Area_Y_STD', 14:'Cont_Area_Z_Mean', 15:'Cont_Area_Z_STD', 16:'Cont_Area_Normal_X_Mean',
                                17:'Cont_Area_Normal_X_STD', 18:'Cont_Area_Normal_Y_Mean', 19:'Cont_Area_Normal_Y_STD',
                                20:'Cont_Area_Normal_Z_Mean', 21:'Cont_Area_Normal_Z_STD', 22:'Cont_Area_Size_Mean',
                                23:'Cont_Area_Size_STD', 24:'Strain to failure', 25:'Maximal Stress',
                                26:'Thoughness', 27:'Initial Effective Stiffness'})
    return df

def correlation_matrix(df):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm

    fig = plt.figure()
    cmap = cm.get_cmap('jet', 30)
    cax = plt.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    plt.title('Abalone Feature Correlation')
    plt.grid(True)
    labels=['Orientation STD', 'Length STD','Diameter STD','SVF Mean X','SVF STD X',
                            'SVF Mean Y', 'SVF STD Y', 'SVF Mean Z', 'SVF STD Z', 'Realsurface', 'Cont_Area_X_Mean', 'Cont_Area_X_STD',
                            'Cont_Area_Y_Mean', 'Cont_Area_Y_STD', 'Cont_Area_Z_Mean', 'Cont_Area_Z_STD', 'Cont_Area_Normal_X_Mean',
                            'Cont_Area_Normal_X_STD', 'Cont_Area_Normal_Y_Mean', 'Cont_Area_Normal_Y_STD',
                            'Cont_Area_Normal_Z_Mean', 'Cont_Area_Normal_Z_STD', 'Cont_Area_Size_Mean',
                            'Cont_Area_Size_STD', 'Strain to failure', 'Maximal Stress',
                            'Thoughness', 'Initial Effective Stiffness']
    plt.xlabel(labels)
    plt.ylabel(labels)
    fig.colorbar(cax)
    plt.show()


np_file=np.load('dict_statab_final_10_03_20.npy',allow_pickle=True).item()
df=Convert2Pandas(np_file)
df=df.astype(float)
df=df[~np.isnan(df).any(axis=1)]

D_strain = df['Strain to failure']
D_stress = df['Maximal Stress']
D_Thoughness = df['Thoughness']
Stiffness =df['Initial Effective Stiffness']


## Some plottings

##############################How Many Simulation Convergence #####################################
#a1,M1=Mean_Val_Con(D_strain)
#a1.savefig('MeanStrain-NOS.png')
#a2,M2=Mean_Val_DS=Mean_Val_Con(D_Thoughness)
#a2.savefig('MeanThoughness-NOS.png')
#a3,M3=Mean_Val_DT=Mean_Val_Con(D_stress)
#a3.savefig('MeanStress-NOS.png')
#a4,M4=Mean_Val_Con(Stiffness)
#a4.savefig('MeanStiffness-NOS.png')

#STD_Val_Con(D_strain)
#STD_Val_Con(D_Thoughness)
#STD_Val_Con(D_stress)
#STD_Val_Con(Stiffness)
##############################How Many Simulation Convergence #####################################



#######################################Correlation#################################

#df_cleand=df.drop(columns=['Realsurface','SVF Mean X','SVF Mean Z','SVF Mean Y', 'SVF STD Y', 'SVF STD X'
#                    ,'SVF STD Z','Cont_Area_Normal_Y_STD','Cont_Area_Normal_Z_Mean','Cont_Area_Normal_X_STD','Cont_Area_Z_Mean','Cont_Area_Size_STD',],axis=1)
#df_cleaned_corr=df_cleand.corr(method='pearson')
#df_cleaned_corr=df.corr(method='pearson')

#correlation_matrix(df_cleaned_corr)


#############################################save##
#df.to_pickle('./df.pkl')

###################################################################################

#x = df.values #returns a numpy array
#x_scaled=preprocessing.scale(x)

#min_max_scaler = preprocessing.MinMaxScaler()
#x_scaled = min_max_scaler.fit_transform(x)
#df_scaled = pd.DataFrame(x_scaled,columns=['Orientation STD','Length STD','Diameter STD','SVF Mean X','SVF STD X',
#                            'SVF Mean Y','SVF STD Y','SVF Mean Z','SVF STD Z','Realsurface','Cont_Area_X_Mean','Cont_Area_X_STD',
#                            'Cont_Area_Y_Mean','Cont_Area_Y_STD','Cont_Area_Z_Mean','Cont_Area_Z_STD', 'Cont_Area_Normal_X_Mean',
#                            'Cont_Area_Normal_X_STD','Cont_Area_Normal_Y_Mean','Cont_Area_Normal_Y_STD',
#                            'Cont_Area_Normal_Z_Mean','Cont_Area_Normal_Z_STD','Cont_Area_Size_Mean',
#                            'Cont_Area_Size_STD','Strain to failure','Maximal Stress',
#                            'Thoughness','Initial Effective Stiffness'])
#df_scaled_corr = df_scaled.corr
#

#g = sns.pairplot(df_scaled, vars=['Strain to failure','Maximal Stress',
#                            'Thoughness','Initial Effective Stiffness'],hue='Orientation STD',kind='reg' )
#g = sns.pairplot(df_scaled, vars=['Strain to failure','Maximal Stress',
#                            'Thoughness','Initial Effective Stiffness'],hue='Orientation STD',kind='reg' )

#fig, axs = plt.subplots(nrows=2, ncols=2)
#f, (ax1, ax2) = plt.subplots(2)

#import importlib
#importlib.reload(plt); importlib.reload(sns)
#sns.set_style("white")

#g1=sns.jointplot('Orientation STD','Strain to failure', data=df, kind="reg",joint_kws = {'scatter_kws':dict(alpha=0.2)})

#g2=sns.jointplot('Orientation STD','Initial Effective Stiffness', data=df, kind="reg",joint_kws = {'scatter_kws':dict(alpha=0.2)})
#
#g3=sns.jointplot('Orientation STD','Thoughness', data=df, kind="reg",joint_kws = {'scatter_kws':dict(alpha=0.2)})
#
#g4=sns.jointplot('Orientation STD','Maximal Stress', data=df, kind="reg",joint_kws = {'scatter_kws':dict(alpha=0.2)})

#plt.close('all')
#
#g1=sns.jointplot('Length STD','Strain to failure', data=df, kind="reg")

#g2=sns.jointplot('Length STD','Initial Effective Stiffness', data=df, kind="reg")

#g3=sns.jointplot('Length STD','Thoughness', data=df, kind="reg")

#g4=sns.jointplot('Length STD','Maximal Stress', data=df, kind="reg")
##
#plt.close('all')

#g1=sns.jointplot('Diameter STD','Strain to failure', data=df, kind="reg")

#g2=sns.jointplot('Diameter STD','Initial Effective Stiffness', data=df, kind="reg")

#g3=sns.jointplot('Diameter STD','Thoughness', data=df, kind="reg")

#g4=sns.jointplot('Diameter STD','Maximal Stress', data=df, kind="reg")

#plt.close('all')
##
#g1=sns.jointplot('Cont_Area_Size_Mean','Strain to failure', data=df, kind="reg",joint_kws = {'scatter_kws':dict(alpha=0.2)})
#
#g2=sns.jointplot('Cont_Area_Size_Mean','Initial Effective Stiffness', data=df, kind="reg",joint_kws = {'scatter_kws':dict(alpha=0.2)})
#
#g3=sns.jointplot('Cont_Area_Size_Mean','Thoughness', data=df, kind="reg",joint_kws = {'scatter_kws':dict(alpha=0.2)})
#
#g4=sns.jointplot('Cont_Area_Size_Mean','Maximal Stress', data=df, kind="reg",joint_kws = {'scatter_kws':dict(alpha=0.2)})
#plt.close('all')


#g1=sns.jointplot('Cont_Area_Z_STD','Strain to failure', data=df, kind="reg")
#
#g2=sns.jointplot('Cont_Area_Z_STD','Initial Effective Stiffness', data=df, kind="reg")
#
#g3=sns.jointplot('Cont_Area_Z_STD','Thoughness', data=df, kind="reg")
#
#g4=sns.jointplot('Cont_Area_Z_STD','Maximal Stress', data=df, kind="reg")

#g1=sns.jointplot('Cont_Area_Normal_Z_Mean','Strain to failure', data=df, kind="reg",joint_kws = {'scatter_kws':dict(alpha=0.2)})
#
#g2=sns.jointplot('Cont_Area_Normal_Z_Mean','Initial Effective Stiffness', data=df, kind="reg",joint_kws = {'scatter_kws':dict(alpha=0.2)})
#
#g3=sns.jointplot('Cont_Area_Normal_Z_Mean','Thoughness', data=df, kind="reg",joint_kws = {'scatter_kws':dict(alpha=0.2)})
#
#g4=sns.jointplot('Cont_Area_Normal_Z_Mean','Maximal Stress', data=df, kind="reg",joint_kws = {'scatter_kws':dict(alpha=0.2)})

#plt.close('all')


#g = sns.pairplot(df, vars=['Orientation STD', 'Initial Effective Stiffness'],hue='Orientation STD')
#g1 = sns.pairplot(df, vars=['Length STD', 'Initial Effective Stiffness'],hue='Length STD')
#g2 = sns.pairplot(df, vars=['Diameter STD', 'Initial Effective Stiffness'],hue='Diameter STD')


#g = sns.pairplot(df, vars=['Orientation STD','Thoughness'],hue='Length STD')
#g = sns.pairplot(df, vars=['Orientation STD','Maximal Stress'],hue='Length STD')
#g = sns.pairplot(df, vars=['Orientation STD','Strain to failure'],hue='Length STD')
#g = sns.pairplot(df, vars=['Length STD', 'Strain to failure'])
#g = sns.pairplot(df, vars=['Diameter STD', 'Strain to failure'])
