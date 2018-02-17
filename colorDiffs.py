import sys
import os
from matplotlib import pyplot as pl

vwFullPath = os.path.abspath(".")
print(vwFullPath)
sys.path.append(vwFullPath)

from blenderCol import *

def getColorsDiffs(diffsB, plotTrajParams):

  nrBiomk = diffsB.shape[0]
  colsB = np.zeros((nrBiomk,3), float)

  #minHue = 0
  #maxHue = 0.66
  #avgHue = 0.33

  minHue = 0
  maxHue = 3

  minB = np.min(diffsB)
  maxB = np.max(diffsB)
  print('minB', minB)
  print('maxB', maxB)
  diffNormB = (diffsB - minB) / (maxB - minB) # put it in [0, 1] range

  huesB = minHue + diffNormB * (maxHue - minHue) #put it in [avg, (maxHue-minHue)] range

  ones = np.array([1,1,1])

  for b in range(nrBiomk):  # nr points
    #colsB[b,:] = colorsys.hsv_to_rgb(0, 1, huesB[b])
    colsB[b, :] = huesB[b] * ones
    # print(np.argmax(clustProbBC[b, :]), hue, colsB[b])

  colsBAll = colsB[plotTrajParams['nearestNeighbours'],:]

  pl.hist(diffNormB,bins=20)
  pl.show()

  return colsBAll

painter = CorticalPainter()
painter.prepareScene()

file = os.getenv('file')
pngFile = os.getenv('pngFile')
isCluster = os.getenv('isCluster')

#file = 'resfiles/adniThMo10kCl4_VWDPMLinear/params_o30.npz'
print('loading file %s' % file)
# print(ads)
dataStruct = pickle.load(open(file, 'rb'))

diffsB = dataStruct['diffsB']
plotTrajParams = dataStruct['plotTrajParams']

outFile = pngFile

freesurfPath = getPaths(isCluster)
importMeshes(freesurfPath)
# for o in range(nrOuterIt):

# print('slopesSortedInd[0]',slopesSortedInd[0] )
colsB = getColorsDiffs(diffsB, plotTrajParams)
print(colsB.shape, colsB)
makeSnapshotBlender(outFile, colsB)