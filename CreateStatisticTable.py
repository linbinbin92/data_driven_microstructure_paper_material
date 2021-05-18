#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: b.lin@mfm.tu-darmstadt.de

"""

import os
import glob
import subprocess
import time
import pandas as pd
import matplotlib.pyplot as plt
from numpy import trapz
from scipy.integrate import simps
import time
start_time = time.time()

import os
import glob
import pandas as pd
import numpy as np
from statistics import mean
from statistics import stdev as std
import matplotlib.pyplot as plt

cdir=os.getcwd()

def getSoftPoint(t,F):
    iES=(F[1]-F[0])/(t[1]-t[0])

    for i in range(F.size-1):

        if F[i]>F[i+1]:
           end=i
           MaxStress=F[i]
           FailStrain=t[i]
           Thoughness=trapz(F[0:end],t[0:end])
           print('loop ended')

           break

        else:
           MaxStress=F.iloc[-1]
           FailStrain=t.iloc[-1]
           Thoughness=trapz(F,t)
    return iES,MaxStress,FailStrain,Thoughness

###################### getting contact statistics from csv####################
def AreaContact(IdO,AreaO):
    N=[]

    MaxId=max(IdO);

    for i in range(MaxId):
       idx= np.array(np.argwhere(IdO==i))

       AreaN=sum(AreaO[idx])
       N=np.append(N,AreaN)
    return N

################## Functions reading dictionary values from csv##############
def getdictvalue_coor(dictval):
    values = dictval.values()
    dictval_mean_x=[]
    dictval_std_x=[]
    dictval_mean_y=[]
    dictval_std_y=[]
    dictval_mean_z=[]
    dictval_std_z=[]
    for value in values:
        dictval_mean_x.append(value[0])
        dictval_std_x.append(value[1])
        dictval_mean_y.append(value[2])
        dictval_std_y.append(value[3])
        dictval_mean_z.append(value[4])
        dictval_std_z.append(value[5])
    return dictval_mean_x, dictval_std_x, dictval_mean_y, dictval_std_y, dictval_mean_z, dictval_std_z

def getdictvalue_normal(dictval):
    values = dictval.values()
    dictval_mean_x=[]
    dictval_std_x=[]
    dictval_mean_y=[]
    dictval_std_y=[]
    dictval_mean_z=[]
    dictval_std_z=[]
    for value in values:
        dictval_mean_x.append(value[0])
        dictval_std_x.append(value[1])
        dictval_mean_y.append(value[2])
        dictval_std_y.append(value[3])
        dictval_mean_z.append(value[4])
        dictval_std_z.append(value[5])
    return dictval_mean_x, dictval_std_x, dictval_mean_y, dictval_std_y, dictval_mean_z, dictval_std_z

def getdictvalue_area(dictval):
    # type of dictval must be dictionary
    values = dictval.values()
    dictval_mean=[]
    dictval_std=[]
    for value in values:
        dictval_mean.append(value[0])
        dictval_std.append(value[1])
    return dictval_mean, dictval_std


####################### actual program reading csv from path ##################

fig=plt.figure()
count=0

Statab = np.load('dict_statab.npy',allow_pickle='TRUE').item()

for root, dirs, files in os.walk(cdir):

    for p in Statab:

        try:
            name='FN_'+p
            path = os.path.join(root, name)
            print(name)

            os.chdir(path)
            files=glob.glob("%s/*.csv" %path)

            for file in files:

                    if 'coordinates' in file:
                        count=count+1
                        coord = pd.read_csv(file)
                        x_co=coord['Points_0']
                        y_co=coord['Points_1']
                        z_co=coord['Points_2']

                        Statab[p].append(mean(x_co))
                        Statab[p].append(std(x_co))
                        Statab[p].append(mean(y_co))
                        Statab[p].append(std(y_co))
                        Statab[p].append(mean(z_co))
                        Statab[p].append(std(z_co))

            for file in files:
                    if 'interface' in file:
                        orient = pd.read_csv(file)
                        x_n=orient['GlyphVector_0']
                        y_n=orient['GlyphVector_1']
                        z_n=orient['GlyphVector_2']

                        Statab[p].append(mean(x_n))
                        Statab[p].append(std(x_n))
                        Statab[p].append(mean(y_n))
                        Statab[p].append(std(y_n))
                        Statab[p].append(mean(z_n))
                        Statab[p].append(std(z_n))

                        Rid=np.array(orient['RegionId'])
                        Qual=np.array(orient['Quality'])
                        N=AreaContact(Rid,Qual)
                        Statab[p].append(mean(N))
                        Statab[p].append(std(N))

            csv=name+'_out.csv'
            data = pd.read_csv(csv)
            t=data['time']
            F=data['ReactionForce_front']
            iES,MaxStress,FailStrain,Thoughness = getSoftPoint(t,F)

            Statab[p].append(FailStrain)
            Statab[p].append(MaxStress)
            Statab[p].append(Thoughness)
            Statab[p].append(iES)
            plt.plot(t,F,'-*',label='%s ' % p)

        except:
            pass

fig.legend()
np.save('dict_statab_final.npy',Statab,allow_pickle=True)
