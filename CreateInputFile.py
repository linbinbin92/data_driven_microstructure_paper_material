#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: b.lin@mfm.tu-darmstadt.de

"""


def CreateInputFileUpper(InputFileName,MeshFile):

    data = open(InputFileName,'w+')
    data.write("\n[Mesh]\
    \n  type = FileMesh\
    \n  file = %s \
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
    \n      top_right = '0.4 0 0.052'  #xmax ymin+a zmax\
    \n      # depends_on = 'block1'\
    \n    [../]\
    \n    [./surface2]\
    \n      type = BoundingBoxNodeSet\
    \n      new_boundary = 'w'\
    \n      bottom_left = '0 0 0' # xmin ymin zmin\
    \n      top_right = '0 0.4 0.052' # xmin+a ymax zmax\
    \n      # depends_on = 'block1'\
    \n    [../]\
    \n    [./surface3]\
    \n      type = BoundingBoxNodeSet\
    \n      new_boundary = 'n'\
    \n      bottom_left = '0.0 0.397 0.0' #xmin ymax-a zmin\
    \n      top_right = '0.4  0.4  0.052'  #xmax ymax zmax\
    \n       # depends_on = 'block1'\
    \n     [../]\
    \n     [./surface4]\
    \n      type = BoundingBoxNodeSet\
    \n      new_boundary = 'o'\
    \n      bottom_left = '0.396 0.0 0.0' #xmax-a ymin+a zmin\
    \n      top_right = '0.4 0.4 0.052'  #xmax ymax zmax\
    \n      # depends_on = 'block1'\
    \n      [../]\
    \n   []\
    \n\
    \n[GlobalParams]\
    \n PhiN = 3.61667e-07  #A.Kulachenko T.uesaka MoM (2012)\
    \n PhiT = 5.044e-06  #A.Kulachenko T.uesaka MoM (2012) 3.6275\
    \n MaxAllowableTraction ='0.00206667 0.00646667 0.00646667' # #A.Kulachenko T.uesaka MoM (2012)\
    \n DeltaN = 0.00035 # #A.Kulachenko T.uesaka MoM (2012)\
    \n DeltaT = 0.00156 # #A.Kulachenko T.uesaka MoM (2012)\
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
    \n  [./vonMises]\
    \n    family = MONOMIAL\
    \n    order = FIRST\
    \n  [../]\
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
    \n  [./vonMises]\
    \n    type = RankTwoScalarAux\
    \n    rank_two_tensor = stress\
    \n    variable = vonMises\
    \n    scalar_type = VonMisesStress\
    \n  [../]\
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
    \n[Materials] " %(MeshFile))
    data.close()

def CreateInputFileLower(InputFileName):
        data = open(InputFileName,'w+')
        data.write("\n [../]\
        \n    [./strain]\
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
        \n  nl_rel_tol = 1e-10\
        \n  nl_abs_tol = 1e-09\
        \n  nl_max_its = 30\
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
        \n  print_linear_residuals = true\
        \n  console = true\
        \n  csv = true\
        \n  interval = 3\
        \n  perf_graph = true\
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
        \n  # [Elastic_Energy_sum]\
        \n  #   type = ElementAverageValue\
        \n  #   variable = 'E'\
        \n  # [../]\
        \n[] ")
        data.close()
