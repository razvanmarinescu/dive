import bpy
import random
import colorsys
import numpy as np
import os
import pickle
from abc import ABC, abstractmethod
from socket import gethostname
import sys

# # exec(compile(open('blenderCol.py').read(), 'blenderCol.py', 'exec'))


def getPaths(cluster):
  hostName = gethostname()
  if hostName == 'razvan-Inspiron-5547':
    freesurfPath = '/usr/local/freesurfer-6.0.0'
  elif hostName == 'razvan-Precision-T1700':
    freesurfPath = '/usr/local/freesurfer-6.0.0'
  elif cluster:
    freesurfPath = '/home/rmarines/src/freesurfer-6.0.0'
  else:
    raise ValueError('check hostname or if running on cluster')

  return freesurfPath

class BrainPainter(ABC):

  def prepareScene(self):
    # delete the cube
    scene = bpy.context.scene
    for ob in scene.objects:
      if ob.type == 'MESH' and ob.name.startswith("Cube"):
        ob.select = True
      else:
        ob.select = False
    bpy.ops.object.delete()
    bpy.data.worlds['World'].horizon_color = (1, 1, 1)

    self.setCamera()
    self.setLamp()

  def deletePrevLamps(self):
    scene = bpy.data.scenes["Scene"]
    for key in [k for k in scene.objects.keys() if k.startswith('Lamp')]:
      scene.objects[key].select = True
    bpy.ops.object.delete()

    for lamp_data in bpy.data.lamps:
      bpy.data.lamps.remove(lamp_data)

  def prepareCamera(self):
    scene = bpy.data.scenes["Scene"]

    # Set render resolution
    scene.render.resolution_x = 1200  # resolutions are twice what shown here
    scene.render.resolution_y = 900

    # Set camera fov in degrees
    fov = 50.0
    pi = 3.14159265
    scene.camera.data.angle = fov * (pi / 180.0)
    scene.camera.data.lens = 100

    # Set camera rotation in euler angles
    scene.camera.rotation_mode = 'XYZ'

  @abstractmethod
  def setCamera(self):
    pass

  @abstractmethod
  def setLamp(self):
    pass


class CorticalPainter(BrainPainter):
  def __init__(self, ):
    pass

  def setCamera(self):

    scene = bpy.data.scenes["Scene"]

    self.prepareCamera()

    pi = 3.14159265
    scene.camera.rotation_euler = (pi / 2, pi / 2, -1 * pi / 2)
    # Set camera location
    scene.camera.location = (-167.00, -0.48, 1.824)

    bpy.data.cameras['Camera'].type = 'ORTHO'
    bpy.data.cameras['Camera'].ortho_scale = 220
    bpy.data.cameras['Camera'].clip_end = 1000

  def setLamp(self):

    energyAll = 5
    distanceAll = 1000

    scene = bpy.data.scenes["Scene"]
    self.deletePrevLamps()

    lampIndices = [1, 2, 3, 4]
    lampLocs = [(-136, 45, 72), (-136, -105, -64), (-136, -105, 72), (-136, 45, -64)]
    nrLamps = len(lampIndices)

    for l in range(nrLamps):
      # Create new lamp datablock
      lamp_data = bpy.data.lamps.new(name="lamp%d data" % lampIndices[l], type='POINT')
      # Create new object with our lamp datablock
      lamp = bpy.data.objects.new(name="Lamp%d" % lampIndices[l], object_data=lamp_data)
      # Link lamp object to the scene so it'll appear in this scene
      scene.objects.link(lamp)
      # Place lamp to a specified location
      scene.objects['Lamp%d' % lampIndices[l]].location = lampLocs[l]
      lamp_data.energy = energyAll
      lamp_data.distance = distanceAll


def getInterpColors(clustProbBC, plotTrajParams, clustHuePoints, slopesSortedInd):
  print(np.sum(clustProbBC, 1))
  assert (all(np.abs(np.sum(clustProbBC, 1) - 1) < 0.001))
  print(clustHuePoints.shape)
  print(clustProbBC[0, slopesSortedInd].shape)
  # assert clustProbBC[0, slopesSortedInd].shape[0] == clustHuePoints.shape[0]

  nrBiomk, nrClust = clustProbBC.shape
  print('clustHuePoints', clustHuePoints)

  colsB = np.zeros((nrBiomk,3), float)
  colsC = np.zeros((nrClust,3), float)

  for b in range(nrBiomk):  # nr points
    hue = np.sum(clustHuePoints * clustProbBC[b, slopesSortedInd])
    # print('clustProbBC[b, slopesSortedInd]', clustProbBC[b, slopesSortedInd])
    # print('clustHuePoints', clustHuePoints)
    # print(hue)
    colsB[b,:] = colorsys.hsv_to_rgb(hue, 1, 1)
    # print(np.argmax(clustProbBC[b, :]), hue, colsB[b])

  for c in range(nrClust):
    hue = clustHuePoints[c]
    colsC[c,:] = colorsys.hsv_to_rgb(hue, 1, 1)

  # print(plotTrajParams['nearestNeighbours'].shape, plotTrajParams['nearestNeighbours'])
  colsBAll = colsB[plotTrajParams['nearestNeighbours'],:]

  return colsBAll

def getMaxLikColors(clustProbBC, plotTrajParams, clustHuePoints, slopesSortedInd):
  print(np.sum(clustProbBC, 1))
  assert (all(np.abs(np.sum(clustProbBC, 1) - 1) < 0.001))

  nrBiomk, nrClust = clustProbBC.shape
  print('clustHuePoints', clustHuePoints)

  colsB = np.zeros((nrBiomk,3), float)
  colsC = np.zeros((nrClust,3), float)

  for b in range(nrBiomk):  # nr points
    hue = clustHuePoints[np.argmax(clustProbBC[b, slopesSortedInd])]
    colsB[b,:] = colorsys.hsv_to_rgb(hue, 1, 1)
    # print(np.argmax(clustProbBC[b, :]), hue, colsB[b])

  for c in range(nrClust):
    hue = clustHuePoints[c]
    colsC[c,:] = colorsys.hsv_to_rgb(hue, 1, 1)

  # print(plotTrajParams['nearestNeighbours'].shape, plotTrajParams['nearestNeighbours'])
  colsBAll = colsB[plotTrajParams['nearestNeighbours'],:]

  return colsBAll

def importMeshes(freesurfPath):
  fsaverageInflatedLhObj = '%s/subjects/fsaverage/surf/lh.inflated.obj' % freesurfPath
  bpy.ops.import_scene.obj(filepath=fsaverageInflatedLhObj, use_split_objects=False,
    use_split_groups=False)
  print(bpy.data.objects.keys())

  # fsaverageInflatedLh = '%s/subjects/fsaverage/surf/lh.inflated' % \
  #                       freesurfPath
  # coordsLh, facesLh, _ = nib.freesurfer.io.read_geometry(fsaverageInflatedLh, read_metadata=True)



  # vertIndex = 105680  # (motor cortex)

  # print('coordsLh[vertIndex,:]', coordsLh[:20, :])
  # print(asds)

def makeSnapshotBlender(outFile, colsB):

  # start in object mode
  obj = bpy.data.objects["lh.inflated"]
  for ob in bpy.context.scene.objects:
    if ob.type == 'MESH' and ob.name.startswith("lh.inflated"):
      ob.select = True
    else:
      ob.select = False

  if not obj.data.vertex_colors:
    obj.data.vertex_colors.new()

  """
  let us assume for sake of brevity that there is now
  a vertex color map called  'Col'
  """

  color_layer = obj.data.vertex_colors["Col"]

  # or you could avoid using the color_layer name
  # color_layer = mesh.vertex_colors.active

  # print(obj.data.vertices, len(obj.data.vertices))
  # print('obj.data.vertices[vertIndex].co', [obj.data.vertices[v].co for v in range(20)])
  # print(asds)
  #print(obj.data.polygons, len(obj.data.polygons))
  #print(asds)

  # check ordering is ok

  i = 0
  for poly in obj.data.polygons:
    for loop_index in poly.loop_indices:
      loop_vert_index = obj.data.loops[loop_index].vertex_index
      #print(obj.data.loops[loop_index], loop_vert_index)
      colorCurrVertex = colsB[loop_vert_index,:]
      # colorCurrVertex = [0, 1, 0]
      # if np.abs(loop_vert_index - vertIndex) < 2:
      #   colorCurrVertex = [1, 0, 0]

      color_layer.data[loop_index].color = colorCurrVertex
      i += 1

  # set to vertex paint mode to see the result
  mat = bpy.data.materials.new('material_1')
  obj.active_material = mat
  mat.use_vertex_color_paint = True

  bpy.data.scenes['Scene'].render.filepath = outFile

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

