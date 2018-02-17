import argparse
import os
import sys
from socket import gethostname
import time



# don't change to from voxCommon import * as this could end up importing matplotlib
from voxCommon import addParserArgs

parser = argparse.ArgumentParser(description='Launches voxel-wise/point-wise DPM on ADNI'
                                             'using cortical thickness maps derived from MRI')
addParserArgs(parser)
args = parser.parse_args()

# don't import matplotlib until here, add other imports below
if args.agg:
  # print(matplotlib.__version__)
  import matplotlib
  # print(matplotlib.get_backend())
  matplotlib.use('Agg')
  # print(matplotlib.get_backend())
  # print(asds)

from voxCommon import *
import evaluationFramework, drcDEM, adniDEM
from voxelDPM import *
from aux import *
from adniCommon import *
from env import *
import PlotterVDPM

params, plotTrajParams = initCommonVoxParams(args)

plotTrajParams['legendCols'] = 3
plotTrajParams['diagColors'] = {CTL:'b', EMCI:'y', LMCI:'g', AD:'r', SMC:'m'}
plotTrajParams['diagLabels'] = {CTL:'CTL', EMCI:'EMCI', LMCI:'MCI',
  AD:'AD', SMC:'SMC'}
plotTrajParams['ylimitsRandPoints'] = (-5,5)
plotTrajParams['diagNrs'] = [CTL, AD]

plotTrajParams['SubfigClustMaxWinSize'] = (1300, plotTrajParams['SubfigClustMaxWinSize'][1])
plotTrajParams['Clust3DMaxWinSize'] = (900, 600)
plotTrajParams['ylimTrajWeightedDataMean'] = (-1.6,1)
plotTrajParams['ylimTrajSamplesInOneNoData'] = (-2.5,1.5)
plotTrajParams['biomkAxisLabel'] = 'Cortical Thickness Z-score'

plotTrajParams['biomkWasInversed'] = False

def launchADNIthick(runIndex, nrProcesses, modelToRun):

  # dataStruct['pointIndices'] = np.array(range(dataStruct['lhData'].shape[1]))
  # pickle.dump(dataStruct, open(inputFileData, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
  inputPrefix = 'cortThickADNI3Scans'
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
  visit = infoStruct['visit']
  assert (not np.any(np.isnan(data)))

  print('diag', np.unique(diag), diag)
  # print(adsas)

  idx = [0,1,2,3,4]
  # print('partCode[idx]', partCode[idx])
  # print('ageAtScan[idx]', ageAtScan[idx])
  # print('scanTimepts[idx]', scanTimepts[idx])
  # print('diag[idx]', diag[idx])
  # print('visit[idx]', visit[idx])
  # print(adas)

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
  maxNrZeros = 5
  selectedBiomk = np.sum(data == 0, 0) < maxNrZeros

  # import pdb
  # pdb.set_trace()

  data = data[:,selectedBiomk]
  pointIndices = pointIndices[selectedBiomk]

  # calculate Z-scores at each point w.r.t controls at baseline
  controlBlInd = np.logical_and(diag == CTL, scanTimepts == 1)
  meanCTL = np.mean(data[controlBlInd],0) # calculate Z-scores
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

  data = dataZ
  assert (not np.any(np.isnan(data)))

  dataAD = data[diag == AD, :]
  indMaxAbnormality = np.argsort(np.mean(dataAD, 0))  # lowest cortical thickness
  print(indMaxAbnormality)

  sortedByPvalInd, labels, names = testMeanBiomkValue(data, diag, pointIndices, plotTrajParams)
  #doTtest(data, diag, pointIndices)

  #sortedByPvalInd = sortedByPvalInd[selectedBiomk]
  assert(sortedByPvalInd.shape[0] == data.shape[1])

  print(infoStruct['cogTestsLabels'])
  sys.stdout.flush()

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
  params['datasetFull'] = 'adniThick'
  params['fixSpeed'] = False # if true then don't model progression speed, only time shift

  # map points that have been removed to the closest included points (nearestNeighbours).
  # also find the adjacency list for the MRF and another subset of 10k points for
  # initial clustering
  runPartNN = 'L'
  plotTrajParams['nearestNeighbours'], params['adjList'], \
    params['nearNeighInitClust'], params['initClustSubsetInd'] = findNearNeigh(runPartNN,
    params['datasetFull'], pointIndices, plotTrajParams['freesurfPath'], indMaxAbnormality)
  # print(ads)

  diagNrs = np.unique(diag)
  # print('diagNrs, diag', diagNrs, diag)
  # print(asdas)

  # print(len(params['acqDate']), data.shape[0])
  sys.stdout.flush()
  assert(params['data'].shape[0] == params['diag'].shape[0] ==
    params['scanTimepts'].shape[0] == params['partCode'].shape[0] ==
    params['ageAtScan'].shape[0] == params['cogTests'].shape[0])

  # sets an uninformative or informative prior
  priorNr = setPrior(params, args.informPrior, mean_gamma_alpha=1,
    std_gamma_alpha=0.3, mu_beta=0, std_beta=5)

  expName = 'adniThFWHM%dInit%sCl%dPr%dRa%dMrf%d' % (args.fwhmLevel,
    args.initClustering, params['nrClust'], priorNr, args.rangeFactor, args.alphaMRF)
  plotTrajParams['sortedByPvalInd'] = sortedByPvalInd
  plotTrajParams['pointIndices'] = pointIndices
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
  params['runPartCogCorrMain'] = ['L', 'L', 'I', 'I', 'I']
  params['runPartDirDiag'] = ['R', 'R', 'I']
  params['runPartStaging'] = ['L', 'L', 'I']
  params['runPartDiffDiag'] = ['R', 'R', 'I']
  params['runPartConvPred'] = ['I', 'I', 'I']
  params['runPartCVNonOverlap'] = ['I']
  params['runPartCVNonOverlapMain'] = ['L', 'L', 'I', 'I', 'I']
  params['masterProcess'] = runIndex == 0

  # visRegions(data, diag, ageAtScan, plotTrajParams)
  #
  # visData(data, diag, ageAtScan, plotTrajParams,sortedByPvalInd)
  # print(dsasa)

  # makeAvgBiomkMaps(data, diag, ageAtScan, plotTrajParams,
  #   'adniTh', args.fwhmLevel, plotTrajParams['diagLabels'])
  # print(adsa)

  # (longData, longDiagAllTmpts, longDiag, longScanTimepts, longPartCode, longAgeAtScan,
  #  uniquePartCodeInverse, crossData, crossDiag, scanTimepts, crossPartCode, crossAgeAtScan) = \
  #   createLongData(data, diag, scanTimepts, partCode, ageAtScan)
  #
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

  # print(adsas)

  if params['masterProcess']:
    # [initClust, modelFit, AIC/BIC, blender, theta_sampling]
    params['runPartStd'] = ['R', 'R', 'R', 'R', 'R']
    params['runPartMain'] = ['I', 'I', 'I']  # [mainPart, plot, stage]
    params['runPartCogCorr'] = ['R']
    params['runPartCogCorrMain'] = ['L', 'L', 'I', 'I', 'I']
    params['runPartDirDiag'] = ['R', 'R', 'I']
    params['runPartStaging'] = ['L', 'L', 'I']
    params['runPartDiffDiag'] = ['R', 'R', 'I']
    params['runPartConvPred'] = ['I', 'I', 'I']
    params['runPartCVNonOverlap'] = ['I']
    params['runPartCVNonOverlapMain'] = ['R', 'R', 'I', 'R', 'R']

  runAllExpFunc = adniDEM.runAllExpADNI
  modelNames, res = evaluationFramework.runModels(params, expName, modelToRun, runAllExpFunc)

  if params['masterProcess']:
    printResADNIthick(modelNames, res, plotTrajParams)

    expNameBefCl = 'adniThFWHM%dInit%s' % (args.fwhmLevel, args.initClustering)
    expNameAfterCl = 'Pr%dRa%dMrf%d' % (args.informPrior, args.rangeFactor, args.alphaMRF)
    # nrClustList = range(2, 30)
    #nrClustList = [2,3,4,5,6,7,8,9,10,12,15,18,20,25,30,35,40,50]
    nrClustList = [2,3,4,5,6,7,8,9,10]
    # nrClustList = [2,3]
    #nrClustList = [12, 15, 18, 20, 25, 30, 35, 40, 50]
    # printBICresults(params, modelNames, res, expNameBefCl, expNameAfterCl, modelToRun, nrClustList, runAllExpFunc)


def printResADNIthick(modelNames, res, plotTrajParams):
  #nrModels = len(modelNames)

  # dinamicModelName = 'VWDPMLinear'
  # staticModelName = 'VWDPMLinearStatic'
  # dinamicModelName = 'VDPM_MRF'
  # staticModelName = 'VWDPMStatic'
  # noDPSModelName = 'VDPMNoDPS'

  print('##### biomk prediction ######')
  nrModels = len(modelNames)
  pred = list(range(nrModels))
  predMean = list(range(nrModels))
  predStd = list(range(nrModels))
  for m in range(nrModels):
    pred[m] = res[m]['cogCorr']['predStats']
    predMean[m] = np.nanmean(pred[m])
    predStd[m] = np.nanstd(pred[m])

  for m in range(nrModels):
    print('%s predAllFolds' % modelNames[m], pred[m])
  for m in range(nrModels):
    print('%s predMean' % modelNames[m], predMean[m])
  for m in range(nrModels):
    print('%s predStd' % modelNames[m], predStd[m])

  stats = list(range(nrModels))
  print('##### correlation with cog tests ######')
  for m in range(nrModels):
    stats[m] = res[m]['cogCorr']['statsAllFolds']  # shape (NR_FOLDS, 2*NR_COG_TESTS)
    print('stats:', stats[m])
    print(modelNames[m],end=' ')
    meanStats = np.nanmean(stats[m], 0)
    stdStats = np.nanstd(stats[m], 0)
    for i in range(int(meanStats.shape[0]/2)): # don't show Spearman's coefficients
      print('%.2f +/- %.2f & ' % (meanStats[i], stdStats[i]), end='')
    print('\\\\')

  plotScoresHist(scores = pred, labels=modelNames)

  nrCogStats = stats[0].shape[1]

  # perform paired t-test, as the same cross-validation folds have been used in both cases
  tStats = np.zeros(nrCogStats,float)
  pVals = np.zeros(nrCogStats,float)
  for t in range(nrCogStats):
    tStats[t], pVals[t] = scipy.stats.ttest_rel(stats[0][:,t], stats[0][:,t])

  # expInds = [dinIndex, staIndex, noDPSIndex]
  # printDiffs(expInds, res, modelNames)

  # testPredFPBDin = res[dinIndex]['cogCorr']['testPredPredFPB']
  # testPredFPBSta = res[staIndex]['cogCorr']['testPredPredFPB']
  # testPredFPBNoDPS = res[noDPSIndex]['cogCorr']['testPredPredFPB']
  # testPredDataFPB = res[noDPSIndex]['cogCorr']['testPredDataFPB']
  #
  # outDir = 'resfiles/%s' % plotTrajParams['expName']
  # os.system('mkdir -p %s' % outDir)
  #
  # for i in [0]: #range(len(testPredFPBDin)):
  #   print('testPred diff [%d] ' % i, testPredFPBDin[i] - testPredFPBSta[i])
  #
  #   meanAbsDiffB = np.sum(np.abs(testPredFPBDin[i] - testPredFPBSta[i]), axis=0)
  #   plotter = PlotterVDPM.PlotterVDPM()
  #   plotter.plotDiffs(meanAbsDiffB, plotTrajParams,
  #     filePathNoExt='%s/diffPredDinSta_f%d' % (outDir, i))



# def printDiffs(expInds, res, modelNames):
#
#   for ind in range(len(expInds)):




if __name__ == '__main__':
  # model 4 - VDPM sigmoidal
  # model 5 - VDPM linear

  if args.modelToRun:
    modelToRun = args.modelToRun
  elif args.models:
    modelToRun = np.array([int(i) for i in args.models.split(',')])
  else:
    raise ValueError('need to set either --models or --firstModel & --lastModel')

  launchADNIthick(args.runIndex, args.nrProc, modelToRun)
