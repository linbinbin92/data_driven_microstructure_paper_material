from paraview.simple import *

import os
import glob
import time
start_time = time.time()


cdir=os.getcwd()
print(cdir)
count=0
for root, dirs, files in os.walk(cdir):
   for name in dirs:
      count=count+1
      p= os.path.join(root, name)
      p=os.path.abspath(p)
      print('at location %s ' %p)
      os.chdir('%s' %p)

      files=glob.glob("*.e")
      print(files)
      csvfiles=glob.glob("*.csv")
      print(csvfiles)
#
      if 'coordinates.csv' and 'interface.csv' in csvfiles:
            print('paraview files alreaday exists, skip')
            count=count-1
            continue
      print(files)
      print('file numer is %i' %count)
#        os.
       # create a new 'ExodusIIReader'
      test_2_outee = ExodusIIReader(FileName=['%s' %file])
      test_2_outee.ElementVariables = []
      test_2_outee.PointVariables = []
      test_2_outee.GlobalVariables = []
      test_2_outee.NodeSetArrayStatus = []
      test_2_outee.SideSetArrayStatus = []

    # Properties modified on test_2_outee
      test_2_outee.ElementVariables = ['Damage', 'E']
      test_2_outee.PointVariables = ['Damage', 'E', 'R', 'disp_']
      test_2_outee.GlobalVariables = ['D', 'D0', 'ReactionForce_front']
      test_2_outee.SideSetArrayStatus = ['interface']
      test_2_outee.ElementBlocks = []
      test_2_outee.FilePrefix = ''
      test_2_outee.FilePattern = ''

    # Create a new 'SpreadSheet View'
      spreadSheetView1 = CreateView('SpreadSheetView')

    # show data in view
      test_2_outeeDisplay_1 = Show(test_2_outee, spreadSheetView1)

    # get layout
    # layout2 = GetLayoutByName("Layout #2")
    #
    # # assign view to a particular cell in the layout
    # AssignViewToLayout(view=spreadSheetView1, layout=layout2, hint=0)

    # Properties modified on spreadSheetView1
      spreadSheetView1.HiddenColumnLabels = ['Block Number', 'Point ID', 'Damage', 'E', 'GlobalNodeId', 'PedigreeNodeId', 'Points', 'Points_Magnitude', 'R', 'R_Magnitude', 'disp_', 'disp__Magnitude', 'vonMises']

    # Properties modified on spreadSheetView1
      spreadSheetView1.HiddenColumnLabels = ['Block Number', 'Point ID', 'Damage', 'E', 'GlobalNodeId', 'PedigreeNodeId', 'Points_Magnitude', 'R', 'R_Magnitude', 'disp_', 'disp__Magnitude', 'vonMises']

    # export view
      ExportView('coordinates.csv', view=spreadSheetView1)

    ## create a new 'Merge Blocks'
      mergeBlocks1 = MergeBlocks(Input=test_2_outee)

    # # show data in view
    # mergeBlocks1Display = Show(mergeBlocks1, spreadSheetView1)
    #
    # # hide data in view
    # Hide(test_2_outee, spreadSheetView1)

    #update the view to ensure updated data information
    #spreadSheetView1.Update()

    # create a new 'Connectivity'
      connectivity1 = Connectivity(Input=mergeBlocks1)

    # show data in view
    # connectivity1Display = Show(connectivity1, spreadSheetView1)
    #
    # # hide data in view
    # Hide(mergeBlocks1, spreadSheetView1)

    # # update the view to ensure updated data information
    # spreadSheetView1.Update()

    # create a new 'Mesh Quality'
      meshQuality1 = MeshQuality(Input=connectivity1)

    # Properties modified on meshQuality1
      meshQuality1.TriangleQualityMeasure = 'Area'

    # show data in view
    # meshQuality1Display = Show(meshQuality1, spreadSheetView1)

    # hide data in view
    # Hide(connectivity1, spreadSheetView1)

    # update the view to ensure updated data information
    # spreadSheetView1.Update()

    # create a new 'Normal Glyphs'
      normalGlyphs1 = NormalGlyphs(Input=meshQuality1)

    # show data in view
      normalGlyphs1Display = Show(normalGlyphs1, spreadSheetView1)

    # update the view to ensure updated data information
      spreadSheetView1.Update()

    # Properties modified on spreadSheetView1
    # spreadSheetView1.ColumnToSort = 'Points_0'
    # spreadSheetView1.InvertOrder = 1
    #
    # # Properties modified on spreadSheetView1
    # spreadSheetView1.InvertOrder = 0

    # Properties modified on spreadSheetView1
      spreadSheetView1.HiddenColumnLabels = ['Block Number', 'Point ID', 'Damage', 'E', 'GlobalNodeId', 'PedigreeNodeId', 'Points_Magnitude', 'R', 'R_Magnitude', 'disp_', 'disp__Magnitude', 'vonMises', 'GlyphVector_Magnitude']

    # Properties modified on spreadSheetView1
      spreadSheetView1.HiddenColumnLabels = ['Block Number', 'Point ID', 'Damage', 'E', 'GlobalNodeId', 'PedigreeNodeId', 'Points_Magnitude', 'R', 'R_Magnitude', 'disp_', 'disp__Magnitude', 'vonMises', 'GlyphVector_Magnitude', 'ObjectId']

    # Properties modified on spreadSheetView1
      spreadSheetView1.HiddenColumnLabels = ['Block Number', 'Point ID', 'Damage', 'E', 'GlobalNodeId', 'PedigreeNodeId', 'Points_Magnitude', 'R', 'R_Magnitude', 'disp_', 'disp__Magnitude', 'vonMises', 'GlyphVector_Magnitude', 'ObjectId', 'Points']

    # Properties modified on spreadSheetView1
      spreadSheetView1.HiddenColumnLabels = ['Block Number', 'Point ID', 'Damage', 'E', 'GlobalNodeId', 'PedigreeNodeId', 'Points_Magnitude', 'R', 'R_Magnitude', 'disp_', 'disp__Magnitude', 'vonMises', 'GlyphVector_Magnitude', 'ObjectId', 'Points', 'SourceElementId']

    # Properties modified on spreadSheetView1
      spreadSheetView1.HiddenColumnLabels = ['Block Number', 'Point ID', 'Damage', 'E', 'GlobalNodeId', 'PedigreeNodeId', 'Points_Magnitude', 'R', 'R_Magnitude', 'disp_', 'disp__Magnitude', 'vonMises', 'GlyphVector_Magnitude', 'ObjectId', 'Points', 'SourceElementId', 'SourceElementSide']

    # export view
      ExportView('interface.csv', view=spreadSheetView1)
      del test_2_outee
      del spreadSheetView1


print("--- %s seconds ---" % (time.time() - start_time))
