import sys
import os

vwFullPath = os.path.abspath(".")
print(vwFullPath)
sys.path.append(vwFullPath)

from blenderCol import *


def orderTrajBySlope(thetas):
  slopes = (thetas[:, 0] * thetas[:, 1]) / 4
  clustOrderInd = np.argsort(slopes)  # -0.3 -0.7 -1.2   green -> yelllow -> red
  # print('slopes, clustOrderInd', slopes, clustOrderInd)
  return clustOrderInd

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
thetas = dataStruct['thetas']
nrClust = clustProbBC.shape[-1]


assert len(clustProbBC.shape) == 2
nrOuterIt = 1
thetasCX = thetas
slopes = (thetasCX[:, 0] * thetasCX[:, 1]) / 4
# make colors based on slope, from blue (low slope) to red (high slope)
slopesSortedInd = orderTrajBySlope(thetasCX)  # -0.3 -0.7 -1.2   green -> yelllow -> red

# clustProbBC = clustProbBC.reshape(1, clustProbBC.shape[0],
#   clustProbBC.shape[1])
minHue = 0
maxHue = 0.66
minSlope = np.min(slopes)
maxSlope = np.max(slopes)
clustHuePoints = (slopes - minSlope)/(maxSlope - minSlope)
clustHuePoints /= maxHue
# clustHuePoints = [clustHuePoints]
clustHuePoints = np.linspace(minHue, maxHue, nrClust, endpoint=True)
print('slopes', slopes)

# print('slopesSortedInd', slopesSortedInd)

outFiles = []
if pngFile is not None:
  outFiles = [pngFile]
else:
  outFiles = ['%s/blend0.png' % plotTrajParams['outFolder']]


freesurfPath = getPaths(isCluster)
importMeshes(freesurfPath)

print('slopesSortedInd[0]',slopesSortedInd[0] )
colsB = getMaxLikColors(clustProbBC, plotTrajParams, clustHuePoints,
  slopesSortedInd)
print(colsB.shape, colsB)
makeSnapshotBlender(outFiles[0], colsB)

print('-------- Created snapshot: %s -----------' % outFiles[0])