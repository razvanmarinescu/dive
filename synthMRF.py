import argparse
import os
import sys
from socket import gethostname
import os.path


import nibabel as nib
import copy

parser = argparse.ArgumentParser(description='runs a synthetic data experiment')

parser.add_argument('--buildTiny', action="store_true", help='builds the tiny 10k dataset')

parser.add_argument('--trajFunc', dest="trajFunc", help='lin or sig')

parser.add_argument('--runIndex', dest='runIndex', type=int,
                    default=1,help='index of run instance/process')

parser.add_argument('--nrProc', dest='nrProc', type=int,
                   default=1,help='# of processes')

parser.add_argument('--modelToRun', dest='modelToRun', type=int,
                   help='index of model to run')

parser.add_argument('--expToRun', dest='expToRun', type=int,
                   help='index of experiment to run: 0. all 1. vary clusters 2. vary subjects')

parser.add_argument('--stepToRun', dest='stepToRun', type=int,
                   help='index of step to run: 0. all 1-8 otherwise')

parser.add_argument('--nrOuterIt', dest='nrOuterIt', type=int,
                   help='# of outer iterations to run, for estimating clustering probabilities')

parser.add_argument('--nrInnerIt', dest='nrInnerIt', type=int,
                   help='# of inner iterations to run, for fitting the model parameters and subj. shifts')

parser.add_argument('--nrClust', dest='nrClust', type=int,
                   help='# of clusters to fit')

parser.add_argument('--cluster', action="store_true",
                   help='need to include this flag if runnin on cluster')

parser.add_argument('--initClustering', dest="initClustering", default='hist',
                   help='initial clustering method: k-means or hist')

parser.add_argument('--agg', dest='agg', type=int,
                   help='plot figures without using Xwindows, for use on cluster, not linked to cluster as I need to test it locally first')

parser.add_argument('--rangeFactor', dest='rangeFactor', type=float,
                   help='factor x such that min -= rangeDiff*x/10 and max += rangeDiff*x/10')

parser.add_argument('--informPrior', dest='informPrior', type=int, default=0,
                   help='enables informative prior based on gamma and gaussian dist')

# extra params that are not used, only to be able to use

args = parser.parse_args()

if args.agg:
  import matplotlib
  matplotlib.use('Agg')
  # print(asds)

demFullPath = os.path.abspath("../diffEqModel/")
if args.cluster:
  demFullPath = os.path.abspath("../../diffEqModel/")
#print(demFullPath)

voxFullPath = os.path.abspath("../")
sys.path.append(demFullPath)
sys.path.append(voxFullPath)

#sys.path.append("/home/razvan/phd_proj/diffEqModel")
import evaluationFramework, adniDEM
from synthCommon import *
import VDPM_MRF, VDPMMean

hostName = gethostname()
if hostName == 'razvan-Inspiron-5547':
  freesurfPath = '/usr/local/freesurfer-5.3.0'
  homeDir = '/home/razvan'
  blenderPath = 'blender'
elif hostName == 'razvan-Precision-T1700':
  freesurfPath = '/usr/local/freesurfer-5.3.0'
  homeDir = '/home/razvan'
  blenderPath = 'blender'
elif args.cluster:
  freesurfPath = '/share/apps/freesurfer-5.3.0'
  homeDir = '/home/rmarines'
  blenderPath = '/share/apps/blender-2.75/blender'
else:
  raise ValueError('Wrong hostname. If running on new machine, add '
                   'application paths in python code above')

params = {}
params['nrOuterIter'] = args.nrOuterIt
params['nrInnerIter'] = args.nrInnerIt
print(params)
# print(adsa)

rowsColsList = [(1, 3), (2, 3), (2, 4), (3, 4),(3, 5), (3, 6),
  (4, 6),(4, 7), (4, 8), (5,9)]
nrImgMaxList = [x[0]*x[1] for x in rowsColsList]


nrClust = args.nrClust
potentialRowsInd = [j for j in range(len(nrImgMaxList))
                      if nrImgMaxList[j] >= nrClust] + [len(nrImgMaxList) - 1]
# print(potentialRowsList)
nrRows, nrCols = rowsColsList[potentialRowsInd[0]]

assert (nrRows * nrCols >= nrClust)

plotTrajParams = {}
plotTrajParams['nrRows'] = nrRows
plotTrajParams['nrCols'] = nrCols
plotTrajParams['diagColors'] = {CTL:'b', AD:'r'}
plotTrajParams['legendCols'] = 2
plotTrajParams['diagLabels'] = {CTL:'CTL', AD:'AD'}
plotTrajParams['freesurfPath'] = freesurfPath
# plotTrajParams['ylimitsRandPoints'] = (-3,2)
plotTrajParams['blenderPath'] = blenderPath

if args.agg:
  plotTrajParams['agg'] = True
else:
  plotTrajParams['agg'] = False

hostName = gethostname()
if hostName == 'razvan-Inspiron-5547':
  height = 350
else: #if hostName == 'razvan-Precision-T1700':
  height = 350

imgSizes = [(800 * s, height *s) for s in np.linspace(1,2,len(rowsColsList))]
plotTrajParams['SubfigClustMaxWinSize'] = imgSizes[potentialRowsInd[0]]
plotTrajParams['SubfigVisMaxWinSize'] = (1300, height)
plotTrajParams['Clust3DMaxWinSize'] = (900, 600)
# plotTrajParams['ylimTrajWeightedDataMean'] = (-2,0.5)

plotTrajParams['clustHuePoints'] = np.linspace(0,1,nrClust,endpoint=False)
plotTrajParams['clustCols'] = [colorsys.hsv_to_rgb(hue, 1, 0.6) for hue in plotTrajParams['clustHuePoints']]
plotTrajParams['legendColsClust'] = min([nrClust, 4])


def launchSynth(runIndex, nrProcesses, modelToRun):

  runAllExpFunc = runAllExpSynth

  #if os.path.isfile(inputFileData):
  trajFuncDict = {'lin': linearFunc, 'sig': sigmoidFunc}

  # forceRegenerate = False
  forceRegenerate = True

  ############# define default parameters #####################################

  np.random.seed(1)
  nrSubjDef = 300
  nrBiomk = 10000
  # not used directly, relevant for when I use real data as I can map then to the actual freesurfer vertices
  nrClustToGenDef = 3 # number of clusters to generate data from
  nrClustToFit = args.nrClust
  nrTimepts = 4
  trajFunc = trajFuncDict[args.trajFunc]

  lowerAgeLim = 40
  upperAgeLim = 80
  dpsLowerLimit = 0
  dpsUpperLimit = 1
  dpsInterval = dpsUpperLimit - dpsLowerLimit
  ageInterval = upperAgeLim - lowerAgeLim

  avgStdScaleFactor = 1

  slopeLowerLim = -3
  slopeUpperLim = -3
  slopeInterval = slopeUpperLim - slopeLowerLim

  trajMinLowerLim = -1
  trajMinUpperLim = -1
  trajMinInterval = trajMinUpperLim - trajMinLowerLim

  covPerturbed13 = np.diag([0, 0.35, dpsInterval/70,0])
  covPerturbed2 = np.diag([0, 0.1, dpsInterval / 70, 0])
  covPerturbed = [covPerturbed13, covPerturbed2, covPerturbed13]

  covSubjShifts = np.array([[0.05, 0], [0, 10]])  # +/- 10 years shifts on avg, averate rate 1+/-0.4

  makeThetaIdentifFunc = VoxelDPM.makeThetasIdentif

  params['nearNeighInitClust'] = np.array(range(nrBiomk))
  params['initClustSubsetInd'] = np.array(range(nrBiomk))

  ############### set parameters ###############################################

  params['runIndex'] = runIndex
  params['nrProcesses'] = nrProcesses
  params['modelToRun'] = modelToRun
  params['cluster'] = args.cluster
  params['biomkDir'] = DECR
  params['initClustering'] = 'k-means'
  params['rangeFactor'] = float(args.rangeFactor)
  params['pointIndices'] = np.array(range(nrBiomk), int)
  params['alphaMRF'] = 5 # mrf alpha parameter

  plotTrajParams['sortedByPvalInd'] = range(nrBiomk)
  plotTrajParams['pointIndices'] = params['pointIndices']
  plotTrajParams['labels'] = np.zeros(nrBiomk, int)
  plotTrajParams['names'] = ['v']

  params['masterProcess'] = runIndex == 0


  # makes changes to params
  setPrior(params, args.informPrior) # sets an informative or uninformative prior

  nrSteps = 4
  if args.stepToRun == 0:
    stepsList = range(nrSteps)
  else:
    stepsList = [args.stepToRun-1]

  # copy state of params and plotTrajParams
  paramsLocal = copy.deepcopy(params)
  plotTrajParamsLocal = copy.deepcopy(plotTrajParams)
  paramsLocal['plotTrajParams'] = plotTrajParamsLocal

  if args.expToRun == 1 or args.expToRun == 0:
    nrSubjList = [75,50,35,20]
    for i in stepsList:
      expFolder = 'synth/mrfNrSubj%d' % i
      os.system('mkdir -p resfiles/%s' % expFolder)
      nrSubjCurr = nrSubjList[i]
      expNameShort = 'data'
      dataFileName = 'resfiles/%s/%s.npz' % (expFolder, expNameShort)
      paramsLocal['dataset'] = expNameShort
      paramsLocal['datasetFull'] = 'synth%s' % expNameShort

      thetasTrueCurr = generateThetas(nrClustToGenDef, trajMinLowerLim,
        trajMinInterval, slopeLowerLim, slopeInterval, dpsLowerLimit, dpsInterval)

      covPerturbedCurr = [np.diag([0, thetasTrueCurr[c3, 1] ** 2 / 15, dpsInterval / 70, 0])
        for c3 in range(nrClustToGenDef)]

      clustAssignTrueB, thetasPerturbed, paramsLocal['adjList'] = genClustAssThetasPerturb(
        nrBiomk, nrClustToGenDef, thetasTrueCurr, covPerturbedCurr)

      paramsLocal = generateClustData(nrSubjCurr, nrBiomk, nrClustToGenDef,
        nrTimepts, trajFunc, thetasTrueCurr, thetasPerturbed, clustAssignTrueB,
        lowerAgeLim, upperAgeLim, covSubjShifts, avgStdScaleFactor,
        dataFileName, forceRegenerate, makeThetaIdentifFunc, paramsLocal)

      paramsLocal['nrClust'] = paramsLocal['trueNrClust']
      expName = '%s/init%sCl%dPr%dRa%d' % (expFolder,
        args.initClustering, nrClustToFit, args.informPrior, args.rangeFactor)
      plotTrajParams['expName'] = expName
      paramsLocal['plotTrajParams'] = plotTrajParamsLocal

      if modelToRun == 8:
        # VDPM_MRF mean model, without MRF  (Markos Random Field)
        dpmBuilder = VDPMMean.VDPMMeanBuilder(params['cluster'])  # disease progression model builder
        dpmBuilder.setPlotter(PlotterVDPM.PlotterVDPMSynth())
        modelName = 'VDPMMean'
        paramsLocal['currModel'] = 0
        expNameCurrModel = '%s_%s' % (expName, modelName)
        # [initClust, modelFit, AIC/BIC, blender, postFitAnalysis]
        paramsLocal['runPartStd'] = ['L', 'Non-enforcing', 'I', 'I', 'I']
        # [mainPart, plot, stage]
        paramsLocal['runPartMain'] = ['R', 'I', 'I']
        runAllExpFunc(paramsLocal, expNameCurrModel, dpmBuilder,
          compareTrueParamsFunc=compareWithTrueParams)

      if modelToRun == 9:
        # VDPM_MRF mean model, with MRF (Markov Random Field)
        dpmBuilder = VDPM_MRF.VDPMMrfBuilder(params['cluster'])  # disease progression model builder
        dpmBuilder.setPlotter(PlotterVDPM.PlotterVDPMSynth())
        modelName = 'VDPM_MRF'
        paramsLocal['currModel'] = 1
        expNameCurrModel = '%s_%s' % (expName, modelName)
        # [initClust, modelFit, AIC/BIC, blender, postFitAnalysis]
        paramsLocal['runPartStd'] = ['L', 'Non-enforcing', 'I', 'I', 'I']
        # [mainPart, plot, stage]
        paramsLocal['runPartMain'] = ['R', 'I', 'I']
        runAllExpFunc(paramsLocal, expNameCurrModel, dpmBuilder,
          compareTrueParamsFunc=compareWithTrueParams)

def genClustAssThetasPerturb(nrBiomk, nrClust, thetasTrue, covPerturbed):
  '''
  Generates cluster assignment and biomarker trajectories on a 2D grid,
    for testing the MRF. The grid is divided in a 1 x nrClust subgrid
    (nrClust columns), with each subgrid corrresponding to a cluster

  :param nrBiomk:
  :param nrClust:
  :param thetasTrue:
  :param covPerturbed:
  :return:
  '''

  nrRows = int(np.sqrt(nrBiomk))
  assert nrRows**2 == nrBiomk
  nrCols = nrRows
  clustWidthCols = nrCols/nrClust

  adjList = np.zeros((nrBiomk, 4), int) # adjacency list for the MRF
  clustAssignTrueB = np.zeros(nrBiomk, int)
  thetasPerturbed = np.zeros((nrBiomk,4), float)
  biomkIndex = np.zeros((nrRows,nrCols),int)

  biomkCnt = 0 # biomarker counter from 0 to nrBiomk
  for r in range(nrRows):
    for c in range(nrCols):

      clustAssignTrueB[biomkCnt] = int(np.floor(c / clustWidthCols))
      thetasPerturbed[biomkCnt,:] = np.random.multivariate_normal(
        thetasTrue[clustAssignTrueB[biomkCnt], :],
        covPerturbed[clustAssignTrueB[biomkCnt]])


      biomkIndex[r,c] = biomkCnt

      biomkCnt += 1


  # find adjacency list
  biomkCnt = 0  # biomarker counter from 0 to nrBiomk
  for r in range(nrRows):
    for c in range(nrCols):
      currList = []
      if (r - 1) >= 0:
        currList += [biomkIndex[(r - 1),c]]
      else: # if boundary condition then add the other neighbour twice
        currList += [biomkIndex[(r + 1), c]]

      if (r + 1) < nrRows:
        currList += [biomkIndex[(r + 1), c]]
      else:
        currList += [biomkIndex[(r - 1), c]]

      if (c - 1) >= 0:
        currList += [biomkIndex[r, (c - 1)]]
      else:
        currList += [biomkIndex[r, (c + 1)]]

      if (c + 1) < nrCols:
        currList += [biomkIndex[r, (c + 1)]]
      else:
        currList += [biomkIndex[r, (c - 1)]]

      adjList[biomkCnt,:] = currList

      biomkCnt += 1

  return clustAssignTrueB, thetasPerturbed, adjList

def compareWithTrueParams(dpmObj, resStruct):

  import itertools

  print(dpmObj.clustProb)
  trueParams = dpmObj.params['trueParams']
  nrBiomk, nrClust = dpmObj.clustProb.shape
  avgCorrectProb = np.zeros(math.factorial(nrClust), float)
  permSet = list(itertools.permutations(range(nrClust)))
  for p, perm in enumerate(permSet):
    print('permutation', perm)
    clustProbPermed = dpmObj.clustProb[:, perm]
    avgCorrectProb[p] = np.mean([clustProbPermed[b,trueParams['clustAssignB'][b]] for b in range(nrBiomk)])

  inferredPermInd = np.argmax(avgCorrectProb)
  inferredPerm = permSet[inferredPermInd]

  print('inferredPerm', inferredPerm)
  print('avgCorrectProb', avgCorrectProb[inferredPermInd])

  shiftL1mean = np.mean(np.abs(dpmObj.subShifts - trueParams['subShiftsLong']), axis=0)
  subShiftsNaive = np.zeros(dpmObj.subShifts.shape)
  subShiftsNaive[:,0] = 1
  # print(resStruct['ageFirstVisitLong1array'].shape)
  # print(subShiftsNaive.shape)
  subShiftsNaive, _ = dpmObj.makeShiftsIdentif(
    subShiftsNaive, resStruct['ageFirstVisitLong1array'], resStruct['longDiag'])

  # lNorms = np.abs(dpmObj.subShifts - trueParams['subShiftsLong'])

  dpsLongTrue = VoxelDPM.calcDps(trueParams['subShiftsLong'], resStruct['ageFirstVisitLong1array'])
  dpsLong = VoxelDPM.calcDps(dpmObj.subShifts, resStruct['ageFirstVisitLong1array'])
  fig = dpmObj.plotterObj.plotSubShiftsTrue(dpmObj.subShifts,
    trueParams['subShiftsLong'], dpsLong, dpsLongTrue, plotTrajParams,
    replaceFigMode=False, fontsize = 20)
  fig.savefig('%s/synShiftsRes_%s.png' % (dpmObj.outFolder,
    dpmObj.params['plotTrajParams']['outFolder'].split('/')[-1]), dpi=100)
  print(adsa)

  shiftL1meanNaive = np.mean(np.abs(subShiftsNaive - trueParams['subShiftsLong']), axis=0)

  print('shiftL1mean', shiftL1mean)
  print('shiftL1meanNaive', shiftL1meanNaive)
  print('trueParams[subShiftsLong][:10]', trueParams['subShiftsLong'][:10])
  print('dpmObj.subShifts[:10]', dpmObj.subShifts[:10])
  print('subShiftsNaive[:10]', subShiftsNaive[:10])

  print('resStruct[dpsCross][4*4:5*4]', resStruct['dpsCross'][4*4:5*4])

  clustProbPermed = dpmObj.clustProb[:, inferredPerm]
  thetasPermed = dpmObj.thetas[inferredPerm, :]

  ssdThEst = np.zeros(nrClust,float)
  ssdThTrue = np.zeros(nrClust, float)
  ssdThDerivEst = np.zeros((nrClust,thetasPermed.shape[1]), float)
  ssdThDerivTrue = np.zeros((nrClust,thetasPermed.shape[1]), float)

  # see if the lik of the true thetas is higher than that of the estimated params
  for k in range(nrClust):
    print(thetasPermed.shape)
    print(clustProbPermed.shape)
    ssdThEst[k] = VoxelDPM.objFunTheta(dpmObj, thetasPermed[k,:], resStruct['crossData'],
                                     resStruct['dpsCross'], clustProbPermed[:,k])[1]
    ssdThTrue[k] = VoxelDPM.objFunTheta(dpmObj, trueParams['thetas'][k,:], resStruct['crossData'],
                                      resStruct['dpsCross'], clustProbPermed[:,k])[1]

  inferredPermInv = np.argsort(inferredPerm)

  dpsCrossTrue = resStruct['crossAgeAtScan'] * trueParams['subShiftsCross'][:, 0] + \
                 trueParams['subShiftsCross'][:, 1]  # disease progression score
  clustProbTrue = makeClustProbFromArray(trueParams['clustAssignB'])
  clustProbTrueBCColNorm = clustProbTrue / np.sum(clustProbTrue, 0)[None, :]

  plotterObj = PlotterVDPM.PlotterVDPMSynth()

  trueThetasPerturbedClustPerm = [trueParams['thetasPerturbedClust'][i] for i in  inferredPermInv]
  fig = plotterObj.plotTrajWeightedDataMeanTrueParams(resStruct['crossData'], resStruct['crossDiag'],
     dpsCrossTrue, dpmObj.thetas, trueParams['variances'][inferredPermInv],
     clustProbTrueBCColNorm[:, inferredPermInv], dpmObj.params['plotTrajParams'],
     dpmObj.params['trajFunc'], trueParams['thetas'][inferredPermInv, :], showConfInt=False,
     trueThetasPerturbedClust=trueThetasPerturbedClustPerm, replaceFigMode=False, colorTitle=False, fontsize = 16)
  fig.savefig('%s/synThetaRes_%s.png' % (dpmObj.outFolder, dpmObj.outFolder.split('/')[-1]), dpi=100)

  # the algo minimises the objFunc, so the likThTrue should be lower than likThEst
  print('ssdThEst', ssdThEst)
  print('ssdThTrue', ssdThTrue)
  print('ssdThDerivEst', ssdThDerivEst)
  print('ssdThDerivTrue', ssdThDerivTrue)

  # import pdb
  # pdb.set_trace()

  resComp = {}
  resComp['avgCorrectProb'] = avgCorrectProb
  return resComp

if __name__ == '__main__':
  # model 4 - VDPM sigmoidal
  # model 5 - VDPM linear

  if args.modelToRun:
    modelToRun = args.modelToRun
  elif args.models:
    modelToRun = np.array([int(i) for i in args.models.split(',')])
  else:
    raise ValueError('need to set either --models or --firstModel & --lastModel')

  launchSynth(args.runIndex, args.nrProc, modelToRun)



