#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 16:59:15 2019

@author: binbin
"""


import os
import glob
import subprocess
import time
import pandas as pd
import matplotlib.pyplot as plt

# bashCommand = "sbatch job.sh"
# def Submit(dest):

cdir=os.getcwd()
fig=plt.figure()
count=0
for root, dirs, files in os.walk(cdir):

   for name in dirs:
       try:
          pp = os.path.join(root, name)
          os.chdir(pp)
          files=glob.glob("FN_*.csv" )
          data = pd.read_csv(files[0])
          t=data['time']
          F=data['ReactionForce_front']
          plt.plot(t,F,'-',label='%s ' % files)
          count=count+1
          if t.max() > 0.0019 and F.max() > 0.0003: # check outliers
              print(name)
          if  F.max() > 0.0015: # check outliers
              print(name)

       except:
           pass
           
print(count)
# fig.legend()
