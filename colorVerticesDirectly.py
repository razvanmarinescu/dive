import sys
import os

print('------Starting----------')

vwFullPath = os.path.abspath(".")
print(vwFullPath)
sys.path.append(vwFullPath)

# import aux
from blenderCol import *

painter = CorticalPainter()
painter.prepareScene()

file = os.getenv('file')
pngFile = os.getenv('pngFile')
backSideToo = os.getenv('backSideToo')
# file='/home/razvan/phd_proj/voxelwiseDPM/resfiles/fsavgPatches/blender_epicenter_adniPetInitk-meansCl20Pr0Ra1Mrf5DataNZ_VDPM_MRF.npz'
# pngFile='/home/razvan/phd_proj/voxelwiseDPM/resfiles/fsavgPatches/blender_epicenter_adniPetInitk-meansCl20Pr0Ra1Mrf5DataNZ_VDPM_MRF.png'
isCluster = False

#file = 'resfiles/adniThMo10kCl4_VWDPMLinear/params_o30.npz'
print('loading file %s' % file)
# print(ads)
dataStruct = pickle.load(open(file, 'rb'))

vertexCols = dataStruct['vertexCols']
freesurfPath = getPaths(isCluster)
importMeshes(freesurfPath)

makeSnapshotBlender(pngFile, vertexCols)

print('backSideToo', backSideToo)
if backSideToo is not None:
  print('Printing also the back side')
  bpy.context.scene.objects['lh.inflated'].rotation_euler = (1.57, 3.1415, 0)

  # makeSnapshotBlender(, vertexCols)
  backSidePicture = '%s_back.png' % pngFile[:-4]

  bpy.data.scenes['Scene'].render.filepath = backSidePicture

  logfile = 'blender_render.log'
  open(logfile, 'a').close()
  old = os.dup(1)
  sys.stdout.flush()
  os.close(1)
  os.open(logfile, os.O_WRONLY)

  bpy.ops.render.render(write_still=True)

  os.close(1)
  os.dup(old)
  os.close(old)


