# data_driven_microstructure_paper_material

This repository contains the files that have been developed in the publication : [Data-driven microstructure sensitivity study of fibrous paper material] by Lin et. al. Doi: (https://doi.org/10.1016/j.matdes.2020.109193)

1. The developed procedure starts from creating random fibernetwork structures within software Geodict, after generation of the fibernetwork structures and saving the .stl files, this script also incorperates steps as meshing the structures by automatically calling the free meshing tool Gmsh, and write out a input file for running FEM simulation on software MOOSE. Some utils scripts are CreateInputFile, CreateInputFile2, write_mesh_geo... etw. 

2. After generation of the mesh file, the feature characteristics are extracted partly by the free visualization software Paraview, partly inside Geodict. The feature data are summerized in a .pkl file.

3. The acutal calculation has been carried out on a HPC-cluster system. The neccecary scripts running cluster simulation are in the CreateJobFileClusterAndSubmit.py

4. The results of the cohesive mechanical simulation are saved partly as .csv and collected inside CreateStatisticTable.py. Together with input features from previous data, that form the base for the statistical and ML analysis later on.

5. Regression.py loads the neccesary data for the regression inference prediction using machine learning models and outputs the regression results.

6. Permutation.py and HowmanySimulationAreSufficient.py load the neccesary data and performs the sensitivity study.


Disclaimer: This information should provide a guidline of reproduction of the results obtained in the mentioned publication. Due to complexity and different softwares used, no guarantee are provided. However if you have discovered any bugs, errors or for any questions, please adress to the contact avaiable in the publication.  

# Citation
If you use this repository for a publication, then please cite it using the following bibtex-entry:
```
@article{LIN2021109193,
title = {Data-driven microstructure sensitivity study of fibrous paper materials},
journal = {Materials & Design},
volume = {197},
pages = {109193},
year = {2021},
issn = {0264-1275},
doi = {https://doi.org/10.1016/j.matdes.2020.109193},
url = {https://www.sciencedirect.com/science/article/pii/S0264127520307280},
```
