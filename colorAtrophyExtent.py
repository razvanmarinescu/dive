import sys
import os

vwFullPath = os.path.abspath(".")
print(vwFullPath)
sys.path.append(vwFullPath)

from blenderCol import *

painter = CorticalPainter()
painter.prepareScene()

file = os.getenv('file')
pngFile = os.getenv('pngFile')
isCluster = os.getenv('isCluster')

#file = 'resfiles/adniThMo10kCl4_VWDPMLinear/params_o30.npz'
print('loading file %s' % file)
# print(ads)
dataStruct = pickle.load(open(file, 'rb'))

clustProbBC = dataStruct['clustProbBC']
plotTrajParams = dataStruct['plotTrajParams']
biomkValuesThresh = dataStruct['biomkValuesThresh']
nrClust = clustProbBC.shape[-1]

# def plotTrajWeightedDataMean(self, data, diag, dps, longData, longDiag, longDPS, thetas, variances, clustProbBCColNorm, plotTrajParams,
#                                trajFunc, replaceFigMode=True, thetasSamplesClust=None, showConfInt=True,
#                                colorTitle=True, yLimUseData=False, adjustBottomHeight=0.25, orderClust=False):

assert len(clustProbBC.shape) == 2
nrOuterIt = 1

# clustProbBC = clustProbBC.reshape(1, clustProbBC.shape[0],
#   clustProbBC.shape[1])
minHue = 0
maxHue = 0.66
minVal = np.min(biomkValuesThresh)
maxVal = np.max(biomkValuesThresh)
clustHuePoints = (biomkValuesThresh - minVal) / (maxVal - minVal)
clustHuePoints /= maxHue

outFile = pngFile

freesurfPath = getPaths(isCluster)
importMeshes(freesurfPath)
# for o in range(nrOuterIt):

colsB = getInterpColors(clustProbBC, plotTrajParams, clustHuePoints,
  range(nrClust))
print(colsB.shape, colsB)
makeSnapshotBlender(outFile, colsB)
# print(asdsa)