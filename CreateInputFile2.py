#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: b.lin@mfm.tu-darmstadt.de
"""

def CreateInputFileEulerangles (InputFileName, object ,Phi,Theta,Psi):
        data = open(InputFileName,'w')
        data.write("\n[./B%d]\
           \n    type = ComputeElasticityTensor\
           \n    euler_angle_1 = %s\
           \n    euler_angle_2 = %s\
           \n    euler_angle_3 = %s\
           \n    block = %d\
         \n [../] " %(object, Phi, Theta,Psi,object))
        data.close()
