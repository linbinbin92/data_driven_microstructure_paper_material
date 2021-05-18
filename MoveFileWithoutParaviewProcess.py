#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: b.lin@mfm.tu-darmstadt.de
"""

import os
import glob
import shutil

cdir=os.getcwd()
list=[]

## move obtained files from paraview to the processing folder##

for root, dirs, files in os.walk(cdir):
   for name in dirs:
      p= os.path.join(root, name)
      p=os.path.abspath(p)
      os.chdir('%s' %p)
#      print('at location %s ' %p)
      files=glob.glob("*.csv")
#      print(files)
      if 'coordinates.csv' and 'interface.csv' in files:
          print(p)
          list1.append(p)

print(list)

## specify destiantion
destination='/..'
for i in list:
    source=i
    dest = shutil.move(source, destination)

print('finish')
