import argparse
import os
import sys
from socket import gethostname
import time

# don't change to from voxCommon import * as this could end up importing matplotlib
from voxCommon import addParserArgs

parser = argparse.ArgumentParser(description='Launches voxel-wise/point-wise DPM on '
                                             'ADNI using PET images')
addParserArgs(parser)
args = parser.parse_args()

# don't import matplotlib until here, add other imports below
if args.agg:
  # print(matplotlib.__version__)
  import matplotlib
  matplotlib.use('Agg')
  # print(asds)

from voxCommon import *
import evaluationFramework, drcDEM, adniDEM
from voxelDPM import *
from aux import *
from adniCommon import *
from env import *

params, plotTrajParams = initCommonVoxParams(args)

plotTrajParams['legendCols'] = 3
plotTrajParams['diagColors'] = {CTL:'b', EMCI:'y', LMCI:'g', AD:'r', SMC:'m'}
plotTrajParams['diagLabels'] = {CTL:'CTL', EMCI:'EMCI', LMCI:'LMCI',
  AD:'AD', SMC:'SMC'}
plotTrajParams['ylimitsRandPoints'] = (-5,5)

plotTrajParams['Clust3DMaxWinSize'] = (900, 600)
# plotTrajParams['ylimTrajWeightedDataMean'] = (-1.6,2)
plotTrajParams['ylimTrajSamplesInOneNoData'] = (-2.5,1.5)
plotTrajParams['biomkAxisLabel'] = 'SUVR'
plotTrajParams['biomkAxisLabelCV'] = 'SUVR'
plotTrajParams['biomkWasInversed'] = True

def launch(runIndex, nrProcesses, modelToRun):

  # dataStruct['pointIndices'] = np.array(range(dataStruct['lhData'].shape[1]))
  # pickle.dump(dataStruct, open(inputFileData, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

  inputPrefix = 'av45FWHM%dADNI' % args.fwhmLevel
  inputFileDataFull = '../data/ADNI/%sData.npz' % inputPrefix

  inputFileInfo = '../data/ADNI/%sInfo.npz' % inputPrefix
  print(inputFileInfo)
  sys.stdout.flush()
  #if os.path.isfile(inputFileInfo):
  infoStruct = pickle.load(open(inputFileInfo, 'rb'))

  print('will enter readDataFile')
  dataStruct = readDataFile(inputFileDataFull, args.cluster)

  #selectedBiomk = np.array([x for x in range(4,144)])

  # filter AD subjects
  # diagInd = np.array(np.where(matData['diag'] == PCA)[0])
  print('compiling parameters')
  sys.stdout.flush()
  data = dataStruct['avghData']
  diag = np.array(np.squeeze(infoStruct['diag']), int)
  scanTimepts = np.squeeze(infoStruct['scanTimepts'])
  partCode = np.squeeze(infoStruct['partCode'])
  ageAtScan = np.squeeze(infoStruct['ageAtScan'])
  pointIndices = dataStruct['pointIndices']
  cogTests = infoStruct['cogTests']

  assert (not np.any(np.isnan(data)))

  #np.set_printoptions(threshold = np.inf)
  #print(dataZ, np.min(dataZ))
  #print(asdsa)
  #np.set_printoptions(threshold = 3)

  unqPartCode = np.unique(partCode)
  nrUnqPart = len(unqPartCode)

  #print(partCode)
  #print(scanTimepts)
  #print(nrUnqPart)

  #print(np.sum(data == 0, 0))

  # remove some of the vertices/voxels
  # 1. who have many values of zero (suggesting faulty FS estimation
  # /alignment with average)
  # 2. who have very high values (again for mitigating bad FS estimation)
  # (for PET necessary because of SUVR normalisation)
  maxNrZeros = 5
  selectedBiomk = np.sum(np.abs(data) < 0.0001, axis=0) < maxNrZeros
  sortedMaxData1D = np.sort(np.max(data,axis=0))
  maxBiomkVal = sortedMaxData1D[int(sortedMaxData1D.shape[0]*98/100)]
  biomkThreshInd = np.sum(data < maxBiomkVal,axis=0) == data.shape[0]
  selectedBiomk = np.logical_and(selectedBiomk, biomkThreshInd)

  print('initial', selectedBiomk.shape, 'survived', np.sum(selectedBiomk))
  # print(asdsa)

  # import pdb
  # pdb.set_trace()

  data = data[:,selectedBiomk]
  pointIndices = pointIndices[selectedBiomk]

  # calculate Z-scores at each point w.r.t controls at baseline
  controlBlInd = np.logical_and(diag == CTL, scanTimepts == 1)
  meanCTL = np.mean(data[controlBlInd],0)
  stdCTL = np.std(data[controlBlInd],0)
  dataZ = (data - meanCTL[None,:])/stdCTL[None,:]

  meanAgeCTL = np.mean(ageAtScan[controlBlInd],0)
  stdAgeCTL = np.std(ageAtScan[controlBlInd],0)
  ageAtScanZ = (ageAtScan - meanAgeCTL)/stdAgeCTL

  (rowInd, colInd) = np.where(np.isnan(dataZ))

  rowIndUnq = np.unique(rowInd)
  colIndUnq = np.unique(colInd)

  print(rowIndUnq, colIndUnq)
  print(np.where(stdCTL == 0))
  print(data.shape)
  sys.stdout.flush()

  # data = -dataZ
  data = -data
  assert (not np.any(np.isnan(data)))

  sortedByPvalInd, labels, names = testMeanBiomkValue(data, diag, pointIndices, plotTrajParams)
  #doTtest(data, diag, pointIndices)
  # print(adsa)

  dataAD = data[diag == AD, :]
  indSortedAbnorm = np.argsort(np.mean(dataAD, 0))  # lowest cortical thickness
  print(indSortedAbnorm)

  #sortedByPvalInd = sortedByPvalInd[selectedBiomk]
  assert(sortedByPvalInd.shape[0] == data.shape[1])

  print(infoStruct['cogTestsLabels'])
  sys.stdout.flush()
  # print(adass)

  params['data'] = data
  params['diag'] = diag
  params['scanTimepts'] = scanTimepts
  params['partCode'] = partCode
  params['ageAtScan'] = ageAtScanZ
  params['biomkDir'] = DECR
  params['modelToRun'] = modelToRun
  params['cogTests'] = np.squeeze(cogTests) # CDRSOB, ADAS13, MMSE, RAVLT
  params['cogTests'][:,[2,3]] *= -1 # make MMSE and RAVLT have increasing scores from CTL->AD
  # params['acqDate'] = infoStruct['acqDate']
  params['datasetFull'] = 'adniPet'
  params['fixSpeed'] = False # if true then don't model progression speed, only time shift

  runPartNN = 'L'
  plotTrajParams['nearestNeighbours'], params['adjList'], \
    params['nearNeighInitClust'], params['initClustSubsetInd'] = findNearNeigh(
    runPartNN, params['datasetFull'], pointIndices, plotTrajParams['freesurfPath'], indSortedAbnorm)
  # print(adsa)

  sys.stdout.flush()
  assert(params['data'].shape[0] == params['diag'].shape[0] ==
    params['scanTimepts'].shape[0] == params['partCode'].shape[0] ==
    params['ageAtScan'].shape[0] == params['cogTests'].shape[0])

  priorNr = setPrior(params, args.informPrior, mean_gamma_alpha=1,
    std_gamma_alpha=0.3, mu_beta=0, std_beta=5)

  expName = 'adniPetInit%sCl%dPr%dRa%dMrf%dDataNZ' % \
            (args.initClustering, params['nrClust'], priorNr,
            args.rangeFactor, args.alphaMRF)
  plotTrajParams['pointIndices'] = pointIndices
  plotTrajParams['sortedByPvalInd'] = sortedByPvalInd
  plotTrajParams['labels'] = labels
  plotTrajParams['names'] = names
  plotTrajParams['expName'] = expName
  plotTrajParams['ageTransform'] = (meanAgeCTL, stdAgeCTL)
  plotTrajParams['datasetFull'] = params['datasetFull']

  params['plotTrajParams'] = plotTrajParams

  # [initClust, modelFit, AIC/BIC, blender, theta_sampling]
  params['runPartStd'] = ['L', 'Non-enforcing', 'I', 'R', 'L']
  params['runPartMain'] = ['I', 'I', 'I']  # [mainPart, plot, stage]
  params['runPartCogCorr'] = ['R']
  params['runPartCogCorrMain'] = ['R', 'R', 'I', 'R', 'R']
  params['runPartDirDiag'] = ['R', 'R', 'I']
  params['runPartStaging'] = ['L', 'L', 'I']
  params['runPartDiffDiag'] = ['R', 'R', 'I']
  params['runPartConvPred'] = ['I', 'I', 'I']
  params['runPartCVNonOverlap'] = ['I']
  params['runPartCVNonOverlapMain'] = ['R', 'R', 'I', 'R', 'R']
  params['masterProcess'] = runIndex == 0

  # print('data[1,:5000:]', data[1,:5000:])
  # print('diag[:50]', diag[:50])
  # print('ageAtScan[:50]', ageAtScan[:50])

  #visRegions(data, diag, ageAtScan, plotTrajParams)
  #
  #visData(data, diag, ageAtScan, plotTrajParams,sortedByPvalInd)

  # makeAvgBiomkMaps(data, diag, ageAtScan, plotTrajParams,
  #   'adniPet', args.fwhmLevel, plotTrajParams['diagLabels'])

  # (longData, longDiagAllTmpts, longDiag, longScanTimepts, longPartCode, longAgeAtScan,
  #  uniquePartCodeInverse, crossData, crossDiag, scanTimepts, crossPartCode, crossAgeAtScan) = \
  #   createLongData(data, diag, scanTimepts, partCode, ageAtScan)

  # unqDiag = np.unique(longDiag)
  # nrScans = np.zeros(longDiag.shape, float)
  # nrSubjLong = longDiag.shape[0]
  # for s in range(nrSubjLong):
  #   nrScans[s] = longData[s].shape[0]
  #
  # longAgeAtBlScan = np.array([longAgeAtScan[s][0] for s in range(nrSubjLong)])
  #
  # for d in range(unqDiag.shape[0]):
  #   print('%s nrSubj %d' % (plotTrajParams['diagLabels'][unqDiag[d]],
  #     np.sum(longDiag == unqDiag[d], axis=0)))
  #   print('%s nrScans %f' % (plotTrajParams['diagLabels'][unqDiag[d]],
  #     np.mean(nrScans[longDiag == unqDiag[d]])))
  #   print('%s ageAtBlScan %f' % (plotTrajParams['diagLabels'][unqDiag[d]], np.mean(longAgeAtBlScan[longDiag == unqDiag[d]])))
  #
  # print(adsas)

  if params['masterProcess']:
    # [initClust, modelFit, AIC/BIC, blender, theta_sampling]
    params['runPartStd'] = ['L', 'Non-enforcing', 'L', 'I', 'R']
    params['runPartMain'] = ['I', 'I', 'I']  # [mainPart, plot, stage]
    params['runPartCogCorr'] = ['R']
    params['runPartCogCorrMain'] = ['L', 'L', 'I', 'L', 'L']
    params['runPartCVNonOverlap'] = ['I']
    params['runPartCVNonOverlapMain'] = ['L', 'L', 'I', 'R', 'R']

  runAllExpFunc = adniDEM.runAllExpADNI
  modelNames, res = evaluationFramework.runModels(params, expName, modelToRun,
    runAllExpFunc)

  if params['masterProcess']:
    import adniThick

    adniThick.printResADNIthick(modelNames, res, plotTrajParams)

    expNameBefCl = 'adniPetInit%s' % args.initClustering
    expNameAfterCl = 'Pr%dRa%dMrf%dDataNZ' % (args.informPrior, args.rangeFactor,
      args.alphaMRF)
    nrClustList = range(2, 100)
    nrClustList = [2,3,4,5,6,7,8,9,10,12,15,18,19,20,21,22,23,24,25,26,27,28,29,30,
      32,34,36,38,40,50,60,70,80]
    # printBICresults(params, modelNames, res, expNameBefCl, expNameAfterCl, modelToRun, nrClustList, runAllExpFunc)




if __name__ == '__main__':
  # model 4 - VDPM sigmoidal
  # model 5 - VDPM linear

  if args.modelToRun:
    modelToRun = args.modelToRun
  elif args.models:
    modelToRun = np.array([int(i) for i in args.models.split(',')])
  else:
    raise ValueError('need to set either --models or --firstModel & --lastModel')

  launch(args.runIndex, args.nrProc, modelToRun)
