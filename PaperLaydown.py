"""
@author: b.lin@mfm.tu-darmstadt.de
"""

from write_mesh_geo import WriteMesh
from CreateInputFile import CreateInputFileUpper, CreateInputFileLower
from CreateInputFile2 import CreateInputFileEulerangles
from CreateJobFileClusterAndSubmit import CreaJobFileCluster, Submit
import subprocess
import signal
import glob
import time
import os
import stringmap
import numpy as np
import csv
import shutil
import random

gmshpath = './gmsh4' ## speficy the meshing tool path

####### Create fibers##
num_fiber = 40 ## specify the number
for i in range(num:_fiber):

    p1=random.randint(5,45)

    p2=random.uniform(0,5)
    p2=round(p2,1)*1e-5
    p2 = float(format(p2,'.1e'))

    p3=random.uniform(0,5)
    p3=round(p3,1)*1e-6
    p3 = float(format(p3,'.1e'))

    std_orient=p1
    std_len=p2
    std_dia=p3

    Header =  {
      'Release'      : '2020',
      'Revision'     : '36964',
      'BuildDate'    : '11 Oct 2019',
      'CreationDate' : '16 Oct 2019',
      'CreationTime' : '15:15:18',
      'Creator'      : 'binbin',
      'Platform'     : '64 bit Linux',
      }

    Description = '''
    Macro file for GeoDict 2020
    recorded at 15:15:18 on 2019Oct16
    by Binbin Lin of MFM/FB11/TU Darmstadt
    '''
    Variables = {
      'NumberOfVariables' : 3,
     'Variable1' : {
       'Name'           : 'std_orient',
       'Label'          : 'std_orientation',
       'Type'           : 'int',
       'Unit'           : '°',
       'ToolTip'        : 'StandardDeviation of Orientan angle Gaussian distribution .',
       'BuiltinDefault' : 10.0,
       'Check'          : 'min0;max100'
       },
        'Variable2' : {
        'Name'           : 'std_len',
        'Label'          : 'std_length',
        'Type'           : 'double',
        'Unit'           : 'Micron',
        'ToolTip'        : 'StandardDeviation of Fiberlength Gaussian distribution .',
        'BuiltinDefault' : 10.0,
        'Check'          : 'min0;max100'
        },
        'Variable3' : {
        'Name'           : 'std_dia',
        'Label'          : 'std_diameter',
        'Type'           : 'double',
        'Unit'           : 'Micron',
        'ToolTip'        : 'StandardDeviation of Fiberdiameter Gaussian distribution .',
        'BuiltinDefault' : 17,
        'Check'          : 'min0;max100'
        },
      }

    try:
        os.mkdir(f'FN_{std_orient}_{std_len}_{std_dia}')
    except:
        pass
    macrofolder = gd.getMacroFileFolder()

    LayDown_args_1 = {
      'Domain' : {
        'Material' : {
          'Name'        : 'Air',
          'Type'        : 'Fluid', # Possible values: Fluid, Solid, Porous
          'Information' : '',
          },
        'PeriodicX'         : True,
        'PeriodicY'         : True,
        'PeriodicZ'         : False,
        'OriginX'           : (0, 'm'),
        'OriginY'           : (0, 'm'),
        'OriginZ'           : (0, 'm'),
        'VoxelLength'       : (4e-06, 'm'),
        'DomainMode'        : 'VoxelNumber',   # Possible values: VoxelNumber, Length, VoxelNumberAndLength
        'NX'                : 100,
        'NY'                : 100,
        'NZ'                : 1,
        'OverlapMode'       : 'GivenMaterial', # Possible values: OverlapMaterial, NewMaterial, OldMaterial, GivenMaterial
        'OverlapMaterialID' : 1,
        'NumOverlapRules'   : 0,
        'HollowMaterialID'  : 0,
        },
      'OverlapMaterial' : {
        'Name'        : 'Manual',
        'Type'        : 'Solid',   # Possible values: Fluid, Solid, Porous
        'Information' : 'Overlap',
        },
      'FiberType1' : {
        'Material' : {
          'Name'        : 'Cellulose',
          'Type'        : 'Solid',     # Possible values: Fluid, Solid, Porous
          'Information' : '',
          },
        'DiameterDistribution' : {
          'Type'               : 'Gaussian', # Possible values: Constant, UniformlyInInterval, Gaussian, Table, LogNormal
          'MeanValue' : 1.7e-05,
          'StandardDeviation'  : std_dia,
          'Bound'              : 2*std_dia,
          'CutOffDistribution' : True,
          },
        'LengthDistribution' : {
          'Type'               : 'Gaussian', # Possible values: Constant, UniformlyInInterval, Gaussian, Table, LogNormal
          'MeanValue'          : 3e-4,
          'StandardDeviation'  : std_len,
          'Bound'              : 1.99*std_len,
          'CutOffDistribution' : True,
          },
        'TorsionStartAngleDistribution' : {
          'Type'  : 'Constant', # Possible values: Constant, UniformlyInInterval, Gaussian, Table, LogNormal
          'Value' : 0,
          },
        'RotationAngleDistribution' : {
          'Type'  : 'Constant', # Possible values: Constant, UniformlyInInterval, Gaussian, Table, LogNormal
          'Value' : 0,
          },
        'OrientationAngleDistribution' : {
          'Type'               : 'Gaussian', # Possible values: Constant, UniformlyInInterval, Gaussian, Table, LogNormal
          'MeanValue'          : 0,
          'StandardDeviation'  : std_orient,
          'Bound'              : 2*std_orient,
          'CutOffDistribution' : True,
          },
        'Probability'                   : 1,
        'SpecificWeight'                : 1.5,
        'Type'                          : 'ShortCircularPaperFiberGenerator',
        'RoundedEnd'                    : False,
        'TorsionMaxChange'              : 0,
        'OrientationMode'               : 1,
        'OscillationMode'               : 0,
        'SegmentLength'                 : 5e-05,
        'RandomnessInPlane'             : 0,
        'RandomnessPerpendicular'       : 0,
        'StraightnessInPlane'           : 1,
        'ForceInPlane'                  : 0.2,
        'Stiffness'                     : 1e-05,
        },
      'MaterialMode'             : 'Material',          # Possible values: Material, MaterialID
      'MaterialIDMode'           : 'MaterialIDPerObjectType',# Possible values: MaterialIDPerObjectType, MaterialIDPerMaterial
      'ResultFileName'           : f'PaperLayDown_{std_orient}_{std_len}_{std_dia}.gdr',
      'SaveStructure'            : True,
      'RecordIntermediateResult' : False,
      'SaveGadStep'              : 5,
      'PercentageType'           : 1,
      'RandomSeed'               : 235,
      'NumberOfFiberTypes'       : 1,
      'IsolationDistance'        : 0,
      'Grammage'                 : 30,
      'NumberOfObjects'          : 1000,
      'OverlapMode'              : 'IsolationDistance', # Possible values: AllowOverlap, RemoveOverlap, ForceConnection, IsolationDistance, ProhibitWithExisting, ProhibitOverlap, MatchSVFDistribution
      'StoppingCriterion'        : 'Grammage',          # Possible values: SolidVolumePercentage, NumberOfObjects, Grammage, Density, WeightPercentage, FillToRim, SVP, Number
      'InExisting'               : False,
      'KeepStructure'            : False,
      'DeformFibers'             : True,
      'NumberOfShifts'           : 40,
      'NumberOfRotations'        : 20,
      'NumberOfDeformations'     : 100,
      'ShiftMaxAngle'            : 89,
      'RotationMaxAngle'         : 5,
      }
    gd.runCmd("PaperGeo:LayDown", LayDown_args_1, Header['Release'])
    ##################################################################################################################################################
    Permute_args_1 = {
      'PermuteMode' : 'FlipZ', # Possible values: RotateLeft, RotateRight, FlipXY, FlipXZ, FlipYZ, FlipX, FlipY, FlipZ
      }
    gd.runCmd("ProcessGeo:Permute", Permute_args_1, Header['Release'])

    SaveStatistics_args_1 = {
      'BackgroundMode'        : 'Pore',
      'DirectionX'            : True,
      'DirectionY'            : True,
      'DirectionZ'            : True,
      'AnalyzeLayerThickness' : True,
      'ResultFileName'        : f'1DStatistics_{std_orient}_{std_len}_{std_dia}.gdr',
      'Method'                : 3,
      'WriteFile'             : False,
      }
    SVF = gd.runCmd("MatDict:SaveStatistics", SaveStatistics_args_1, Header['Release'])
    SVF = stringmap.parseGDR(SVF)
    # gdr.push("ResultMap")

    SVFX=(SVF.getList('ResultMap:MaterialID1:SolidVolumeFractionX'))
    SVFY=(SVF.getList('ResultMap:MaterialID1:SolidVolumeFractionY'))
    SVFZ=(SVF.getList('ResultMap:MaterialID1:SolidVolumeFractionZ'))

    SVFX = [float(x) for x in SVFX]
    SVFY = [float(x) for x in SVFY]
    SVFZ = [float(x) for x in SVFZ]

    SVF=[SVFX,SVFY,SVFZ]
    np.save(f'SVF_{std_orient}_{std_len}_{std_dia}', np.array(SVF))

    EstimateSurface_args_1 = {
      'ResultFileName' : f'SurfaceArea_{std_orient}_{std_len}_{std_dia}.gdr',
      'IsPeriodic'     : True,
      'SurfaceMode'    : 'Solid',
      }

    EST=gd.runCmd("PoroDict:EstimateSurface", EstimateSurface_args_1, Header['Release'])
    EST = stringmap.parseGDR(EST)
    EST=(EST.getList('ResultMap:EstimateRealSurface:SurfaceArea'))
    np.save(f'EST_{std_orient}_{std_len}_{std_dia}',np.array(EST))

    AnalyzeObjects_args_1 = {
      'ObjectMode'          : 'UseGAD',        # Possible values: UseGAD, UseVG32
      'AnalyzeContacts'     : True,
      'AnalyzeObjectVolume' : False,
      'GadMode'             : 'UseCurrent',    # Possible values: LoadFromFile, UseCurrent
      'MaterialMode'        : 'AllMaterials',  # Possible values: AllMaterials, OnlyOneMaterial
      'MaterialID'          : 0,
      'OverlapMode'         : 'OverlapObject', # Possible values: OverlapObject, OldObject, NewObject
      'SaveOrientation'     : False,
      'SaveG32'             : False,
      'SaveLES'             : False,
      'GADFileName'         : '',
      'Domain' : {
        'PeriodicX'        : False,
        'PeriodicY'        : False,
        'PeriodicZ'        : False,
        'OriginX'          : (0, 'm'),
        'OriginY'          : (0, 'm'),
        'OriginZ'          : (0, 'm'),
        'VoxelLength'      : (1e-06, 'm'),
        'DomainMode'       : 'VoxelNumber',     # Possible values: VoxelNumber, Length, VoxelNumberAndLength
        'NX'               : 0,
        'NY'               : 0,
        'NZ'               : 0,
        'OverlapMode'      : 'OverlapMaterial', # Possible values: OverlapMaterial, NewMaterial, OldMaterial, GivenMaterial
        'Material' : {
          'Type'        : 'Fluid',
          'Name'        : 'Undefined',
          'Information' : '',
          },
        'HollowMaterialID' : 0,
        },
      'G32FileName'         : '',
      'VoxelLength'         : 1e-06,
      'PeriodicX'           : False,
      'PeriodicY'           : False,
      'PeriodicZ'           : False,
      'Histogram' : {
        'HistogramXAxis' : 'Volume',
        'HistogramYAxis' : 'RelativeCount', # Possible values: RelativeCount, CumulativeCount, RelativeVolume, CumulativeVolume
        'HistogramMode'  : 'Automatic',     # Possible values: Automatic, MinMax
        'MinimalValue'   : (0, 'm^3'),
        'MaximalValue'   : (0, 'm^3'),
        'NumberOfBins'   : 10,
        'XAxisRange'     : 'Automatic',     # Possible values: Automatic, MinMax
        'YAxisRange'     : 'Automatic',     # Possible values: Automatic, MinMax
        'XAxisMinValue'  : (0, 'm^3'),
        'XAxisMaxValue'  : (1, 'm^3'),
        'YAxisMinValue'  : 0,
        'YAxisMaxValue'  : 1,
        },
      'ResultFileName'      : f'AnalyzeObjects_{std_orient}_{std_len}_{std_dia}.gdr',
      }
    gd.runCmd("MatDict:AnalyzeObjects", AnalyzeObjects_args_1, Header['Release'])

    SetExpertSettings_args_1 = {
      'NumberOfParameters' : 1,
      'Key1'               : 'MeshGeo:CreateVoxelMesh:UseIndexImage',
      'Value1'             : 1,
      }
    gd.runCmd("GeoDict:SetExpertSettings", SetExpertSettings_args_1, Header['Release'])

    CreateIndexImage_args_1 = {
      }
    gd.runCmd("GadGeo:CreateIndexImage", CreateIndexImage_args_1, Header['Release'])

    CreateVoxelMesh_args_1 = {
      'AddToCurrent'        : False,
      'Isovalue'            : 29491,
      'STLMode'             : 'MultiMaterial', # Possible values: VoxelSurface, Rounded, Smooth, MultiMaterial
      'AvoidRepeatingEdges' : False,
      'StlShrinkOffset'     : 0.45,
      'PeriodicX'           : False,
      'PeriodicY'           : False,
      'PeriodicZ'           : False,
      'CloseWalls'          : False,
      'MaterialSelection' : {
        'AnalyzeMode' : 'Solid', # Possible values: Pore, Solid, ChosenMaterial, ChosenMaterialIDs, All
        'Material' : {
          'Type' : 'Undefined',
          },
        'MaterialIDs' : 'NONE',
        },
      'Coarsen'             : False,
      'CoarsenSettings' : {
        'Mode'              : 'NumberOfTriangles', # Possible values: NumberOfTriangles, EdgeLength, TargetNumberOfTriangles
        'MinimalLength'     : 0,
        'MaximalLength'     : 0,
        'TriangleReduction' : 50,
        'NumberOfFaces'     : 0,
        'Adaptive'          : False,
        'TargetAspectRatio' : 3,
        },
      }
    gd.runCmd("MeshGeo:CreateVoxelMesh", CreateVoxelMesh_args_1, Header['Release'])

    Smooth_args_1 = {
      'Filter' : 'Taubin', # Possible values: Taubin, Lapalacian, Constraint
      }
    gd.runCmd("MeshGeo:Smooth", Smooth_args_1, Header['Release'])

    SaveTriangulationFile_args_1 = {
      'FileName'           : 'SaveSurfaceMesh.stl',
      'Unit'               : 'mm', # Possible values: m, cm, mm, Micron, nm, Inch, Voxel
      'WriteASCII'         : True,
      'WriteMultipleFiles' : True,
      'ResultFileName'     : f'SaveSurfaceMesh_{std_orient}_{std_len}_{std_dia}.gdr',
      }
    gd.runCmd("GeoDict:SaveTriangulationFile", SaveTriangulationFile_args_1, Header['Release'])

    LoadFile_args_1 = {
      'FileName' : f'AnalyzeObjects_{std_orient}_{std_len}_{std_dia}/Contacts.gdt',
      }
    gd.runCmd("GeoDict:LoadFile", LoadFile_args_1, Header['Release'])

    EDT_args_1 = {
      'ResultFileName'       : f'EuclDistTrans_{std_orient}_{std_len}_{std_dia}.gdr',
      'Signed'               : False,
      'HighRes'              : False,
      'ComponentDist'        : False,
      'PeriodicX'            : True,
      'PeriodicY'            : True,
      'PeriodicZ'            : False,
      'SolidBoundaryX'       : False,
      'SolidBoundaryY'       : False,
      'SolidBoundaryZ'       : False,
      'SaveIntermediateData' : False,
      'BackgroundMode'       : 'Pore',
      }
    gd.runCmd("PoroDict:EDT", EDT_args_1, Header['Release'])

    LoadVolumeField_args_1 = {
      'FileName'         : f'EuclDistTrans_{std_orient}_{std_len}_{std_dia}/EuclDistTrans_{std_orient}_{std_len}_{std_dia}.dst',
      'Mode'             : 'All', # Possible values: Selected, All
      'KeepVolumeFields' : False,
      'KeepCompression'  : 'DecompressIfPossible',# Possible values: DecompressIfPossible, KeepCompression
      }
    gd.runCmd("GeoDict:LoadVolumeField", LoadVolumeField_args_1, Header['Release'])
    #
    EDT = gd.getVolumeField(0)
    np.save(f'EDT_{std_orient}_{std_len}_{std_dia}', np.array(EDT))
    # gd.msgBox(EDT)


    #####################################################################
    LoadStructure= glob.glob(f'PaperLayDown_{std_orient}_{std_len}_{std_dia}/*.gdt')
    LoadFile_args_2 = {
      'FileName' : LoadStructure,
      }
    gd.runCmd("GeoDict:LoadFile", LoadFile_args_2, Header['Release'])

    numObjects = gd.getNumberOfGADObjects()
    WriteMesh(numObjects,f'SaveSurfaceMesh_{std_orient}_{std_len}_{std_dia}')

    os.replace("SaveSurfaceMesh.geo", f'SaveSurfaceMesh_{std_orient}_{std_len}_{std_dia}/SaveSurfaceMesh.geo')
    os.chdir(f'SaveSurfaceMesh_{std_orient}_{std_len}_{std_dia}')


    proc=subprocess.Popen([gmshpath, 'SaveSurfaceMesh.geo'],stdout=subprocess.PIPE, shell=False)
    def CreateVolumeMesh():
        while True:
             check = os.path.isfile(f'SaveSurfaceMesh_{std_orient}_{std_len}_{std_dia}.msh')
             if check == True:
                time.sleep(2)
                proc.terminate()
                break
    CreateVolumeMesh()

    os.chdir(macrofolder)

    data = open(f'FN_{std_orient}_{std_len}_{std_dia}.i','w')
    data.write("\n[Mesh]\
    \n  type = FileMesh\
    \n  file = %s\
    \n  construct_side_list_from_node_list = true\
    \n[]\
    \n[MeshModifiers]\
    \n  [./interface]\
    \n    type = BreakMeshByBlock\
    \n  [../]\
    \n[]\
    \n\
    \n[MeshModifiers]\
    \n    [./surface1]\
    \n      type = BoundingBoxNodeSet\
    \n      new_boundary = 's'\
    \n      bottom_left = '0 0 0' # xmin ymin zmin\
    \n      top_right = '0.4 0 0.2'  #xmax ymin+a zmax\
    \n      # depends_on = 'block1'\
    \n    [../]\
    \n    [./surface2]\
    \n      type = BoundingBoxNodeSet\
    \n      new_boundary = 'w'\
    \n      bottom_left = '0 0 0' # xmin ymin zmin\
    \n      top_right = '0 0.4 0.2' # xmin+a ymax zmax\
    \n      # depends_on = 'block1'\
    \n    [../]\
    \n    [./surface3]\
    \n      type = BoundingBoxNodeSet\
    \n      new_boundary = 'n'\
    \n      bottom_left = '0.0 0.4 0.0' #xmin ymax-a zmin\
    \n      top_right = '0.41  0.41  0.2'  #xmax ymax zmax\
    \n       # depends_on = 'block1'\
    \n     [../]\
    \n     [./surface4]\
    \n      type = BoundingBoxNodeSet\
    \n      new_boundary = 'o'\
    \n      bottom_left = '0.4 0.0 0.0' #xmax-a ymin+a zmin\
    \n      top_right = '0.41 0.41 0.2'  #xmax ymax zmax\
    \n      # depends_on = 'block1'\
    \n      [../]\
    \n   []\
    \n\
    \n[GlobalParams]\
    \n PhiN = 3.61667e-07\
    \n PhiT = 5.044e-06 \
    \n MaxAllowableTraction ='0.00206667 0.00646667 0.00646667' \
    \n DeltaN = 0.00035\
    \n DeltaT = 0.00156\
    \n C_ijkl = '100 0.33 0.33 10.72 4.2 10.72 4.35 4.35 35.97'\
    \n[]\
    \n\
    \n[Variables]\
    \n  [./disp_x]\
    \n    initial_condition = 1e-15\
    \n  [../]\
    \n  [./disp_y]\
    \n    initial_condition = 1e-15\
    \n  [../]\
    \n  [./disp_z]\
    \n    initial_condition = 1e-15\
    \n  [../]\
    \n[]\
    \n\
    \n[AuxVariables]\
    \n # [./vonMises]\
    \n #   family = MONOMIAL\
    \n #   order = FIRST\
    \n # [../]\
    \n  [./Rx]\
    \n    family = LAGRANGE\
    \n    order = FIRST\
    \n  [../]\
    \n  [./Ry]\
    \n    family = LAGRANGE\
    \n    order = FIRST\
    \n  [../]\
    \n  [./Rz]\
    \n    family = LAGRANGE\
    \n    order = FIRST\
    \n  [../]\
    \n  [./E]\
    \n    family = MONOMIAL\
    \n    order = CONSTANT\
    \n  [../]\
    \n  []\
    \n\
    \n[Kernels]\
    \n  [./TensorMechanics]\
    \n    displacements = 'disp_x disp_y disp_z'\
    \n    save_in = 'Rx Ry Rz'\
    \n  [../]\
    \n[]\
    \n\
    \n[InterfaceKernels]\
    \n  [./interface_X]\
    \n    type = CZMInterfaceKernel\
    \n    disp_index = 0\
    \n    variable = disp_x\
    \n    neighbor_var = disp_x\
    \n    disp_1 = disp_y\
    \n    disp_1_neighbor = disp_y\
    \n    disp_2 = disp_z\
    \n    disp_2_neighbor = disp_z\
    \n    boundary = interface\
    \n  [../]\
    \n  [./interface_Y]\
    \n    type = CZMInterfaceKernel\
    \n    disp_index = 1\
    \n    variable = disp_y\
    \n    neighbor_var = disp_y\
    \n    disp_1 = disp_x\
    \n    disp_1_neighbor = disp_x\
    \n    disp_2 = disp_z\
    \n    disp_2_neighbor = disp_z\
    \n    boundary = interface\
    \n  [../]\
    \n  [./interface_Z]\
    \n    type = CZMInterfaceKernel\
    \n    disp_index = 2\
    \n    variable = disp_z\
    \n    neighbor_var = disp_z\
    \n    disp_1 = disp_x\
    \n    disp_1_neighbor = disp_x\
    \n    disp_2 = disp_y\
    \n    disp_2_neighbor = disp_y\
    \n    boundary = interface\
    \n  [../]\
    \n[]\
    \n[AuxKernels]\
    \n  #[./vonMises]\
    \n   # type = RankTwoScalarAux\
    \n    #rank_two_tensor = stress\
    \n   # variable = vonMises\
    \n   # scalar_type = VonMisesStress\
    \n  #[../]\
    \n  [./elastic_energy]\
    \n    type = ElasticEnergyAux\
    \n    variable = E\
    \n  [../]\
    \n[]\
    \n\
    \n[UserObjects]\
    \n  [./CZMObject]\
    \n    type = Exp3DUserObject\
    \n    disp_x = disp_x\
    \n    disp_x_neighbor = disp_x\
    \n    disp_y = disp_y\
    \n    disp_y_neighbor = disp_y\
    \n    disp_z = disp_z\
    \n    disp_z_neighbor = disp_z\
    \n    execute_on = 'LINEAR'\
    \n  [../]\
    \n[]\
    \n\
    \n[Materials] " % f'SaveSurfaceMesh_{std_orient}_{std_len}_{std_dia}.msh')


    numObjects = gd.getNumberOfGADObjects()
    # gd.msgBox(numObjects)
    gadobjects={}
    # create a dictionary with the gadobjects
    for object in range(1,numObjects+1):
        gadobjects[object]=gd.getGADObject(object,'Release')

    for object in range(1,numObjects+1):

        GadAdd_args = {
               'Domain' : {
                 'Material' : {
                   'Name' : 'Air',
                   'Type' : 'Fluid', # Possible values: Fluid, Solid, Porous
                   'Information' : ''
                 },
                 'PeriodicX' : True,
                 'PeriodicY' : True,
                 'PeriodicZ' : False,
                 'OriginX' : (0, 'm'),
                 'OriginY' : (0, 'm'),
                 'OriginZ' : (0, 'm'),
                 'VoxelLength' : (4e-06, 'm'),
                 'DomainMode' : 'VoxelNumber', # Possible values: VoxelNumber, Length, VoxelNumberAndLength, Undefined
                 'NX' : 100,
                 'NY' : 100,
                 'NZ' : 50,
                 'OverlapMode' : 'OverlapMaterial', # Possible values: OverlapMaterial, NewMaterial, OldMaterial, GivenMaterial, Undefined
                 'OverlapMaterialID' : 200,
                 'NumOverlapRules' : 0,
                 'HollowMaterialID' : 0
               },
              'Object1' : gadobjects[object],
              'KeepStructure' : 0,
              'NumberOfObjects' : 1
            }

        gd.runCmd("GadGeo:GadAdd", GadAdd_args, Header['Release'])


        if object == 1:
            try:
                os.mkdir(f'GadOrientation_{std_orient}_{std_len}_{std_dia}')
            except:
                pass
        # ------------------- MatDict:GadOrientation ---------------------
        GadOrientation_args = {
          'ResultFileName' : f'GadOrientation_{std_orient}_{std_len}_{std_dia}.gdr',
          'Type' : 'GeneralCurvedCircularFiber'
        }

        gdrPath = gd.runCmd("MatDict:GadOrientation", GadOrientation_args, Header['Release'])

        gdr = stringmap.parseGDR(gdrPath)

        gdr.push("ResultMap")

        Phi=(gdr.getList('Phi'))[0]
        Theta=(gdr.getList('Theta'))[0]
        Psi=(gdr.getList('Psi'))[0]

        # CreateInputFileEulerangles(f'FN_{std_orient}_{std_len}_{std_dia}.i',object,Phi,Theta,Psi)

        data.write("\n[./B%d]\
                   \n    type = ComputeElasticityTensor\
                   \n    euler_angle_1 = %s\
                   \n    euler_angle_2 = %s\
                   \n    euler_angle_3 = %s\
                   \n    block = %d\
                 \n [../] " %(object, Phi, Theta,Psi,object))

    # data.close()
    data.write("\n[./strain]\
            \n    #type = ComputeFiniteStrain\
            \n    type = ComputeSmallStrain\
            \n    displacements = 'disp_x disp_y disp_z'\
            \n  [../]\
            \n  [./stress]\
            \n    #type = ComputeFiniteStrainElasticStress\
            \n    type = ComputeLinearElasticStress\
            \n  [../]\
            \n  [./CZMMaterial]\
            \n    type = Exp3DMaterial\
            \n    uo_CohesiveInterface = CZMObject\
            \n    IsDebug = 0\
            \n    #outputs = exodus\
            \n    # output_properties = 'Damage DamageN DamageT TractionLocal DispJumpLocal'\
            \n    #output_properties = 'Damage Damage Area AreaElemID AreaNodeID AreaNormal'\
            \n    boundary = interface\
            \n  [../]\
            \n[]\
            \n\
            \n[BCs]\
            \n  [./suy]\
            \n    type = PresetBC\
            \n    variable = disp_y\
            \n    value = 0.0\
            \n    boundary = 's'\
            \n  [../]\
            \n  [./nuy]\
            \n    type = PresetBC\
            \n    variable = disp_y\
            \n    value = 0.0\
            \n    boundary = 'n'\
            \n  [../]\
            \n  [./wux]\
            \n    type = PresetBC\
            \n    variable = disp_x\
            \n    value = 0.0\
            \n    boundary = 'w'\
            \n  [../]\
            \n  [./oux]\
            \n    type = FunctionPresetBC\
            \n    variable = disp_x\
            \n    function = t\
            \n    boundary = 'o'\
            \n  [../]\
            \n[]\
            \n\
            \n[Preconditioning]\
            \n  [./smp]\
            \n    type = SMP\
            \n     full = true\
            \n  [../]\
            \n[]\
            \n[Executioner]\
            \n  type = Transient\
            \n  #solve_type = PJFNK\
            \n  solve_type = NEWTON\
            \n  automatic_scaling = true\
            \n  petsc_options_iname = '-pc_type -ksp_gmres_restart -pc_factor_mat_solver_type'\
            \n  petsc_options_value = ' lu       2000                superlu_dist'\
            \n  compute_scaling_once= true\
            \n  nl_rel_tol = 1e-09\
            \n  nl_abs_tol = 1e-08\
            \n  nl_max_its = 50\
            \n  # [./TimeStepper]\
            \n  #   type = IterationAdaptiveDT\
            \n    dt= 1.0e-5\
            \n  #   optimal_iterations = 10\
            \n  #   growth_factor = 1.3\
            \n  #   cutback_factor = 0.5\
            \n  # [../]\
            \n    num_steps = 10000000\
            \n[]\
            \n[Outputs]\
            \n  console = true\
            \n  csv = true\
            \n  interval = 3\
            \n  [./oute]\
            \n    type = Exodus\
            \n    elemental_as_nodal = true\
            \n    output_material_properties = true\
            \n  #  show_material_properties = 'Damage TractionLocal DispJumpLocal'\
            \n    show_material_properties = 'Damage'\
            \n  [../]\
            \n[]\
            \n# [Debug]\
            \n#   show_var_residual = 'disp_x disp_y disp_z'\
            \n# []\
            \n[Postprocessors]\
            \n  # [./vonMises]\
            \n  #   type = ElementAverageValue\
            \n  #   variable = vonMises\
            \n  #   []\
            \n  [ReactionForce_front]\
            \n    type = NodalSum\
            \n    variable = 'Rx'\
            \n    boundary = 'o'\
            \n  [../]\
            \n  [./D0]\
            \n    type = SideValueIntegralPostProcessor\
            \n    input_value = 1.0\
            \n    boundary = interface\
            \n  [../]\
            \n  [./D]\
            \n    type = SideDamgeFractionPostProcess\
            \n    MateName = 'Damage'\
            \n    pps_name = D0\
            \n    boundary = interface\
            \n  [../]\
            \n   [Elastic_Energy_sum]\
            \n     type = ElementAverageValue\
            \n     variable = 'E'\
            \n   [../]\
            \n[] ")

    os.replace(f'SaveSurfaceMesh_{std_orient}_{std_len}_{std_dia}/SaveSurfaceMesh_{std_orient}_{std_len}_{std_dia}.msh', f'FN_{std_orient}_{std_len}_{std_dia}/SaveSurfaceMesh_{std_orient}_{std_len}_{std_dia}.msh')
    os.replace(f'FN_{std_orient}_{std_len}_{std_dia}.i', f'FN_{std_orient}_{std_len}_{std_dia}/FN_{std_orient}_{std_len}_{std_dia}.i')
