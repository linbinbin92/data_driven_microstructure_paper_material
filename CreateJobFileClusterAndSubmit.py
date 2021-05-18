#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: b.lin@mfm.tu-darmstadt.de
"""


import os
import glob
import subprocess
import time

# bashCommand = "sbatch job.sh"
def CreaJobFileCluster(dest):
    os.chdir(dest)
    cdir_origin=os.getcwd()
    print(cdir_origin)

    for root, dirs, files in os.walk(cdir_origin):
       for name in dirs:
          p= os.path.join(root, name)
          os.chdir(p)
          cdir=os.getcwd()
          files=glob.glob("*.i" )

          for file in files:
                try:
                    os.remove("job.sh")
                except:
                    pass
                data = open('job.sh','w')
                data.write("#!/bin/bash\
                           \n#SBATCH -J" +' '+ file[3:])
                data.write("\n#SBATCH -e ./MATID.err.%J\
                            \n#SBATCH -o ./MATID.out.%J\
                            \n# Please use the complete path details :\
                            \n#SBATCH --exclusive\
                            \n#SBATCH -n 24      # Number of MPI processes\
                            \n###SBATCH -c 24      # Number of CPU cores (OpenMP-Threads) per MPI process\
                            \n#SBATCH --mem-per-cpu=2400  # Main memory in MByte per MPI task\
                            \n#SBATCH -t 5:58:00     # Hours, minutes and seconds, or '#SBATCH -t 10' - only minutes\
                            \n#SBATCH --exclusive\
                            \n#SBATCH -C avx2\
                            \n\
                            \n# -------------------------------\
                            \n# Afterwards you write your own commands, e.g.\
                            \n\
                            \nulimit -c unlimited\
                            \nulimit -s unlimited\
                            \n\
                            \nsource ~/.moose-profile\
                            \n\
                            \nsrun  /home/ac01asac/moose_applications/ppr_x64/ppr-opt -i"  + " " + file + " " + "> moose.log")
                data.close

def Submit(dest):
    cdir=dest
    for root, dirs, files in os.walk(cdir):
       for name in dirs:
          pp = os.path.join(root, name)
          os.chdir(pp)
          print('Current Folder is', pp)

          process = subprocess.Popen(["sbatch", "job.sh"], stdout=subprocess.PIPE)
          output, error = process.communicate()
          print("Job successfully submitted")
