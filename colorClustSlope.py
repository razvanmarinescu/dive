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
thetas = dataStruct['thetas']
nrClust = clustProbBC.shape[-1]

minHue = 0
maxHue = 0.66

# if len(clustProbBC.shape) == 3:
#   thetasOCX = thetas[:,-1,:,:]
#   assert(len(thetasOCX.shape) == 3)
#   slopes = (thetasOCX[-1,:, 0] * thetasOCX[-1,:, 1]) / 4
#   clustProbBC = clustProbBC[-1,:,:]
#
# else:
assert len(clustProbBC.shape) == 2
print(thetas)
thetasCX = thetas[-1,-1,:,:]
# make colors based on slope, from blue (low slope) to red (high slope)
slopes = (thetasCX[:, 0] * thetasCX[:, 1]) / 4
clustProbBC = clustProbBC

  # print('slopesSortedInd', slopesSortedInd)

slopesSortedInd = [np.argsort(slopes)[::-1]]  # -0.3 -0.7 -1.2   green -> yelllow -> red
minSlope = np.min(slopes)
maxSlope = np.max(slopes)
hues = (slopes - minSlope) / (maxSlope - minSlope)
hues *= maxHue
# for the last image, color clusters according to slope value
slopeHuePoints = np.sort(hues)[::-1]
print('slopesOC[-1,:]', slopes)
print('slopeHuePoints', slopeHuePoints)


freesurfPath = getPaths(isCluster)
importMeshes(freesurfPath)
print('slopesSortedInd',slopesSortedInd )
colsB = getInterpColors(clustProbBC, plotTrajParams, slopeHuePoints, slopesSortedInd)
print(colsB.shape, colsB)
makeSnapshotBlender(pngFile, colsB)
