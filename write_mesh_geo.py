#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: b.lin@mfm.tu-darmstadt.de
"""

def WriteMesh(num_f,name):

    inp=open("SaveSurfaceMesh.geo",'w+')
    val = str("SaveSurfaceMesh")


    for i in range(int(num_f)):
     inp.write(" Merge \"%s_%i.stl\"; \n" %(val,i+1))


    for i in range(int(num_f)):
        inp.write("Surface Loop" + "("+ str(i+1) + ")" +  "={" + str(i+1) +"};\n" )
        inp.write("Volume" + "("+ str(i+1) + ")" +  "={" + str(i+1) +"};\n")
        inp.write("Physical Volume" + "("+ str(i+1) + ")" +  "={" + str(i+1) +"};\n")


    inp.write("Mesh 3;\n")
    inp.write("Coherence Mesh;\n")
    inp.write("Mesh.Format = 1;\n")
    inp.write("Save \"%s.msh\";" %name)
    inp.close()
