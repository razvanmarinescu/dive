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

rowsColsList = [(1, 3), (2, 3), (2, 4), (3, 4),(3, 5), (3, 6), (4, 6),(4, 7), (4, 8), (5,9), (8,13)]
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
  height = 450

imgSizes = [(800 * s, height *s) for s in np.linspace(1,2,len(rowsColsList))]
plotTrajParams['SubfigClustMaxWinSize'] = imgSizes[potentialRowsInd[0]]
plotTrajParams['SubfigVisMaxWinSize'] = (1300, height)
plotTrajParams['Clust3DMaxWinSize'] = (900, 600)
# plotTrajParams['ylimTrajWeightedDataMean'] = (-2,0.5)

plotTrajParams['clustHuePoints'] = np.linspace(0,1,nrClust,endpoint=False)
plotTrajParams['clustCols'] = [colorsys.hsv_to_rgb(hue, 1, 1) for hue in plotTrajParams['clustHuePoints']]
plotTrajParams['legendColsClust'] = min([nrClust, 4])



### TODO finish this function ####
def setLocalParamsNrClust(nrClustToFitCurr, plotTrajParamsLocal):
  potentialRowsInd = [j for j in range(len(nrImgMaxList))
                       if nrImgMaxList[j] >= nrClustToFitCurr] + [len(nrImgMaxList) - 1]
  # print(potentialRowsList)
  nrRows, nrCols = rowsColsList[potentialRowsInd[0]]
  # assert (nrRows * nrCols >= nrClustToGenCurr)

  plotTrajParamsLocal['nrRows'] = nrRows
  plotTrajParamsLocal['nrCols'] = nrCols
  plotTrajParamsLocal['SubfigClustMaxWinSize'] = imgSizes[potentialRowsInd[0]]
  plotTrajParamsLocal['clustHuePoints'] = np.linspace(0, 1, nrClustToFitCurr, endpoint=False)
  plotTrajParamsLocal['clustCols'] = [colorsys.hsv_to_rgb(hue, 1, 1) for hue in
    plotTrajParamsLocal['clustHuePoints']]
  plotTrajParamsLocal['legendColsClust'] = min([nrClustToFitCurr, 4])

  return plotTrajParamsLocal


def launchSynth(runIndex, nrProcesses, modelToRun):

  runAllExpFunc = runAllExpSynth

  #if os.path.isfile(inputFileData):
  trajFuncDict = {'lin': linearFunc, 'sig': sigmoidFunc}

  # forceRegenerate = True
  forceRegenerate = False

  ############# define default parameters #####################################

  nrSubjDef = 300
  nrBiomk = 1000
  # not used directly, relevant for when I use real data as I can map then to the actual freesurfer vertices
  nrClustToGenDef = 3 # number of clusters to generate data from
  nrClustToFit = args.nrClust
  nrTimepts = 4
  trajFunc = trajFuncDict['sig']

  lowerAgeLim = 40
  upperAgeLim = 80
  dpsLowerLimit = -1
  dpsUpperLimit = 2
  dpsIntervalDef = dpsUpperLimit - dpsLowerLimit
  ageInterval = upperAgeLim - lowerAgeLim

  avgStdScaleFactor = 1

  ''' fit sigmoidal function for trajectory with params [a,b,c,d] with minimum d, maximum a+d,
  slope a*b/4 and slope maximum attained at center c
  f(s|theta = [a,b,c,d]) = a/(1+exp(-b(s-c)))+d'''
  thetasTrue = np.zeros((nrClustToGenDef, 4), float)
  thetasTrue[0, :] = [1,-3,dpsLowerLimit,-1]  # make lines intersect the Y=0 axis at lowerAgeLim
  thetasTrue[1, :] = [1,-1,dpsLowerLimit + dpsIntervalDef/2,-1]
  thetasTrue[2, :] = [1,-3,dpsLowerLimit + dpsIntervalDef,-1]

  slopeLowerLim = -2
  slopeUpperLim = -2
  slopeInterval = slopeUpperLim - slopeLowerLim

  trajMinLowerLim = -5
  trajMinUpperLim = -5
  trajMinInterval = trajMinUpperLim - trajMinLowerLim

  covPerturbed13 = np.diag([0, 0.35, dpsIntervalDef/70,0])
  covPerturbed2 = np.diag([0, 0.1, dpsIntervalDef / 70, 0])
  covPerturbed = [covPerturbed13, covPerturbed2, covPerturbed13]

  covSubjShifts = np.array([[0.05, 0], [0, 10]])  # +/- 10 years shifts on avg, averate rate 1+/-0.4

  makeThetaIdentifFunc = VoxelDPM.makeThetasIdentif

  ############### set parameters ###############################################

  params['runIndex'] = runIndex
  params['nrProcesses'] = nrProcesses
  params['modelToRun'] = modelToRun
  params['cluster'] = args.cluster
  params['biomkDir'] = DECR
  params['initClustering'] = 'k-means'
  params['rangeFactor'] = float(args.rangeFactor)
  params['pointIndices'] = np.array(range(nrBiomk), int)

  plotTrajParams['sortedByPvalInd'] = range(nrBiomk)
  plotTrajParams['pointIndices'] = params['pointIndices']
  plotTrajParams['labels'] = np.zeros(nrBiomk, int)
  plotTrajParams['names'] = ['v']
  params['plotTrajParams'] = plotTrajParams

  # [initClust, modelFit, AIC/BIC, blender, theta_sampling]
  params['runPartStd'] = ['R', 'R']
  # [mainPart, plot, stage]
  params['runPartMain'] = ['R', 'I', 'I']

  params['masterProcess'] = runIndex == 0

  # assign initClustSubsetInd and nearNeighInitClust
  params['initClustSubsetInd'] = np.array(range(nrBiomk)) # set to identity map
  params['nearNeighInitClust'] = np.array(range(nrBiomk)) # set to identity map

  if params['masterProcess']:
    # [initClust, pointIndices, modelFit, AIC/BIC, checkers/visual]
    params['runPartStd'] = ['L', 'L', 'L', 'I', 'I']
    # [mainPart, plot, stage]
    params['runPartMain'] = ['R', 'I', 'I']

  params['compareTrueParamsFunc'] = compareWithTrueParams

  # makes changes to params
  setPrior(params, args.informPrior) # sets an informative or uninformative prior

  nrSteps = 8
  # print('args.stepToRun', args.stepToRun)
  if args.stepToRun == 0:
    stepsList = list(range(nrSteps))
  else:
    stepsList = [args.stepToRun-1]

  # if runIndex > 0:
  #   stepsList = [runIndex - 1]

  # print(args.expToRun)
  # print(adasd)

  ###################### vary trajectory centers ###############################
  # copy state of params and plotTrajParams
  paramsLocal = copy.deepcopy(params)
  plotTrajParamsLocal = copy.deepcopy(plotTrajParams)
  paramsLocal['plotTrajParams'] = plotTrajParamsLocal
  resList = []

  plotterObj = PlotterVDPM.PlotterVDPMSynth()

  if args.expToRun == 1 or args.expToRun == 0:
    dpsIntervalList = dpsIntervalDef * [5,2,1.5,1,0.7,0.5,0.3,0.1]
    for i in stepsList:
      np.random.seed(1)
      expFolderShort = 'trajCent%d' % i
      expFolder = 'resfiles/synth/%s' % expFolderShort
      os.system('mkdir -p %s' % expFolder)
      expNameShort = 'data'
      dataFileName = '%s/%s.npz' % (expFolder, expNameShort)
      paramsLocal['dataset'] = expNameShort
      paramsLocal['datasetFull'] = 'synth%s' % expNameShort
      dpsIntervalCurr = dpsIntervalList[i]

      thetasTrueCurr = generateThetas(nrClustToGenDef, trajMinLowerLim,
        trajMinInterval, slopeLowerLim, slopeInterval, dpsLowerLimit, dpsIntervalCurr)

      covPerturbedCurr = [np.diag([0, thetasTrueCurr[c3, 1] ** 2 / 15, dpsIntervalDef / 70, 0]) for c3 in
        range(nrClustToGenDef)]

      # generate perturbed traj from clusters for each biomk
      # generate rand clust with uniform prob each

      clustAssignTrueB, thetasPerturbed = genClustAssThetasPerturb(
        nrBiomk, nrClustToGenDef, thetasTrueCurr, covPerturbedCurr)

      paramsLocal = generateClustData(nrSubjDef, nrBiomk, nrClustToGenDef,
        nrTimepts, trajFunc, thetasTrueCurr, thetasPerturbed, clustAssignTrueB,
        lowerAgeLim, upperAgeLim, covSubjShifts, avgStdScaleFactor,
        dataFileName, forceRegenerate, makeThetaIdentifFunc, paramsLocal)

      # for nrClustToFitCurr in range(1, 110):

      nrClustToFitCurr = nrClustToGenDef
      # #############
      # setLocalParamsNrClust(nrClustToFitCurr, plotTrajParamsLocal)  # changes plotTrajParamsLocal
      # assert plotTrajParamsLocal['legendColsClust'] == min([nrClustToFitCurr, 4])
      # #############
      # print('got hereeeeeeeeeeeee')
      # print(adsas)

      paramsLocal['nrClust'] = nrClustToGenDef
      expName = 'synth/%s/init%sCl%dPr%dRa%d' % \
                (expFolderShort, args.initClustering, nrClustToFitCurr, args.informPrior, args.rangeFactor)
      plotTrajParamsLocal['expName'] = expName
      paramsLocal['plotTrajParams'] = plotTrajParamsLocal

      modelNames, res = evaluationFramework.runModels(paramsLocal, expName, modelToRun, runAllExpFunc)
      resList += [res]

    xLabelStr = 'Distance between trajectories'
    voxelCorrectAssignMeanValues = [resList[i][0]['resComp']['voxelCorrectAssignMean']
      for i in range(len(stepsList))]
    voxelCorrectAssignStdValues = [resList[i][0]['resComp']['voxelCorrectAssignStd']
      for i in range(len(stepsList))]
    fig = plotterObj.plotSynthResOneExp(voxelCorrectAssignMeanValues, voxelCorrectAssignStdValues,
      [dpsIntervalList[i] for i in stepsList], xLabelStr)
    fig.savefig('resfiles/synth/correctVertices_trajCent.png', dpi=100)

  ###################### vary number of clusters ###############################
  # copy state of params and plotTrajParams
  paramsLocal = copy.deepcopy(params)
  plotTrajParamsLocal = copy.deepcopy(plotTrajParams)
  paramsLocal['plotTrajParams'] = plotTrajParamsLocal
  resList = []

  if args.expToRun == 2 or args.expToRun == 0:
    nrClustToGenList = [2,3,5,10,15,20,50,100]
    for i in stepsList:
      np.random.seed(1)
      expFolderShort = 'nrClust%d' % i
      expFolder = 'resfiles/synth/%s' % expFolderShort
      os.system('mkdir -p %s' % expFolder)
      nrClustToGenCurr = nrClustToGenList[i]
      expNameShort = 'data'
      dataFileName = '%s/%s.npz' % (expFolder, expNameShort)
      paramsLocal['dataset'] = expNameShort
      paramsLocal['datasetFull'] = 'synth%s' % expNameShort

      potentialRowsIndCurr = [j for j in range(len(nrImgMaxList))
                           if nrImgMaxList[j] >= nrClustToGenCurr] + [len(nrImgMaxList) - 1]
      print(potentialRowsIndCurr)
      nrRowsCurr, nrColsCurr = rowsColsList[potentialRowsIndCurr[0]]
      plotTrajParamsLocal['nrRows'] = nrRowsCurr
      plotTrajParamsLocal['nrCols'] = nrColsCurr
      print('nrRowsCurr', nrRowsCurr)
      print('nrColsCurr', nrColsCurr)
      plotTrajParamsLocal['clustHuePoints'] = np.linspace(0, 1, nrClustToGenCurr, endpoint=False)
      plotTrajParamsLocal['clustCols'] = [colorsys.hsv_to_rgb(hue, 1, 1) for hue in plotTrajParamsLocal['clustHuePoints']]
      plotTrajParamsLocal['legendColsClust'] = min([nrClustToGenCurr, 4])
      print(plotTrajParamsLocal['clustHuePoints'])
      # print(adsa)

      thetasTrueCurr = generateThetas(nrClustToGenCurr, trajMinLowerLim,
        trajMinInterval, slopeLowerLim, slopeInterval, dpsLowerLimit, dpsIntervalDef)

      covPerturbedCurr = [np.diag([0,thetasTrueCurr[c3,1]**2/15, dpsIntervalDef/70, 0]) for c3 in range(nrClustToGenCurr)]

      # print('nrClust', nrClust)
      # print('clustAssignTrueB', clustAssignTrueB)
      # print('thetasTrue', thetasTrue)
      # print('covPerturbed', covPerturbed)
      # print(covPerturbed.shape, covPerturbed[clustAssignTrueB[0]], clustAssignTrueB[0])
      # generate perturbed traj from clusters for each biomk
      # generate rand clust with uniform prob each

      clustAssignTrueB, thetasPerturbed = genClustAssThetasPerturb(
        nrBiomk, nrClustToGenCurr, thetasTrueCurr, covPerturbedCurr)

      # print('nrClustToGenCurr', nrClustToGenCurr)
      # print(adas)

      paramsLocal = generateClustData(nrSubjDef, nrBiomk, nrClustToGenCurr,
        nrTimepts, trajFunc, thetasTrueCurr, thetasPerturbed, clustAssignTrueB,
        lowerAgeLim, upperAgeLim, covSubjShifts, avgStdScaleFactor,
        dataFileName, forceRegenerate, makeThetaIdentifFunc, paramsLocal)


      paramsLocal['nrClust'] = nrClustToGenCurr
      expName = 'synth/%s/init%sCl%dPr%dRa%d' % \
                (expFolderShort, args.initClustering, nrClustToGenCurr, args.informPrior, args.rangeFactor)
      plotTrajParamsLocal['expName'] = expName
      paramsLocal['plotTrajParams'] = plotTrajParamsLocal

      modelNames, res = evaluationFramework.runModels(paramsLocal, expName, modelToRun, runAllExpFunc)
      resList += [res]

    xLabelStr = 'Number of clusters'
    voxelCorrectAssignMeanValues = [resList[i][0]['resComp']['voxelCorrectAssignMean']
      for i in range(len(stepsList))]
    voxelCorrectAssignStdValues = [resList[i][0]['resComp']['voxelCorrectAssignStd']
      for i in range(len(stepsList))]
    fig = plotterObj.plotSynthResOneExp(voxelCorrectAssignMeanValues, voxelCorrectAssignStdValues,
      [nrClustToGenList[i] for i in stepsList], xLabelStr, makeInts=True)
    fig.savefig('resfiles/synth/correctVertices_nrClust.png', dpi=100)


  ###################### vary number of subjects ################################

  # copy state of params and plotTrajParams
  paramsLocal = copy.deepcopy(params)
  plotTrajParamsLocal = copy.deepcopy(plotTrajParams)
  resList = []

  if args.expToRun == 3 or args.expToRun == 0:
    nrSubjList = [1000,500,250,100,75,50,35,20]
    for i in stepsList:
      np.random.seed(1)
      expFolderShort = 'nrSubj%d' % i
      expFolder = 'resfiles/synth/%s' % expFolderShort
      os.system('mkdir -p %s' % expFolder)
      nrSubjCurr = nrSubjList[i]
      expNameShort = 'data'
      dataFileName = '%s/%s.npz' % (expFolder, expNameShort)
      paramsLocal['dataset'] = expNameShort
      paramsLocal['datasetFull'] = 'synth%s' % expNameShort

      thetasTrueCurr = generateThetas(nrClustToGenDef, trajMinLowerLim, trajMinInterval,
        slopeLowerLim, slopeInterval, dpsLowerLimit, dpsIntervalDef)

      covPerturbedCurr = [np.diag([0, thetasTrueCurr[c3, 1] ** 2 / 15, dpsIntervalDef / 70, 0])
        for c3 in range(nrClustToGenDef)]

      clustAssignTrueB, thetasPerturbed = genClustAssThetasPerturb(
        nrBiomk, nrClustToGenDef, thetasTrueCurr, covPerturbedCurr)

      paramsLocal = generateClustData(nrSubjCurr, nrBiomk, nrClustToGenDef, nrTimepts,
        trajFunc, thetasTrueCurr, thetasPerturbed, clustAssignTrueB, lowerAgeLim, upperAgeLim, covSubjShifts,
        avgStdScaleFactor, dataFileName, forceRegenerate, makeThetaIdentifFunc, paramsLocal)

      paramsLocal['nrClust'] = paramsLocal['trueNrClust']
      expName = 'synth/%s/init%sCl%dPr%dRa%d' % \
                (expFolderShort, args.initClustering, nrClustToFit, args.informPrior, args.rangeFactor)
      plotTrajParams['expName'] = expName
      paramsLocal['plotTrajParams'] = plotTrajParamsLocal

      modelNames, res = evaluationFramework.runModels(paramsLocal, expName, modelToRun, runAllExpSynth)
      resList += [res]

    xLabelStr = 'Number of Subjects'
    voxelCorrectAssignMeanValues = [resList[i][0]['resComp']['voxelCorrectAssignMean']
      for i in range(len(stepsList))]
    voxelCorrectAssignStdValues = [resList[i][0]['resComp']['voxelCorrectAssignStd']
      for i in range(len(stepsList))]
    fig = plotterObj.plotSynthResOneExp(voxelCorrectAssignMeanValues, voxelCorrectAssignStdValues,
      [nrSubjList[i] for i in stepsList], xLabelStr, makeInts=True, adjLeft=0.2)
    fig.savefig('resfiles/synth/correctVertices_nrSubj.png', dpi=100)


def genClustAssThetasPerturb(nrBiomk, nrClust, thetasTrue, covPerturbed):
  clustAssignTrueB = np.array(np.floor(nrClust * np.random.rand(nrBiomk)), int)
  thetasPerturbed = np.array([np.random.multivariate_normal(
    thetasTrue[clustAssignTrueB[b], :], covPerturbed[clustAssignTrueB[b]])
    for b in range(nrBiomk)])

  return clustAssignTrueB, thetasPerturbed

def inferPermBrute(dpmObj, trueParams):
  nrBiomk, nrClust = dpmObj.clustProb.shape
  import itertools
  avgCorrectProb = np.zeros(math.factorial(nrClust), float)
  permSet = list(itertools.permutations(range(nrClust)))
  for p, perm in enumerate(permSet):
    # print('permutation', perm)
    clustProbPermed = dpmObj.clustProb[:, perm]
    avgCorrectProb[p] = np.mean([clustProbPermed[b,trueParams['clustAssignB'][b]] for b in range(nrBiomk)])

  inferredPermInd = np.argmax(avgCorrectProb)
  inferredPerm = permSet[inferredPermInd]

  correctAssignVal = avgCorrectProb[inferredPermInd]

  return np.array(inferredPerm), correctAssignVal


from random import randint

def perturbPerm(currPerm):
  nrClust = currPerm.shape[0]
  srcPos = randint(0, nrClust - 1)
  trgPos = randint(0, nrClust - 1)

  newPerm = copy.deepcopy(currPerm)

  auxVal = newPerm[trgPos]
  newPerm[trgPos] = newPerm[srcPos]
  newPerm[srcPos] = auxVal

  # print('newPerm', newPerm, '  currPerm', currPerm)

  return newPerm

def inferPermGreedySearch(dpmObj, trueParams):
  nrBiomk, nrClust = dpmObj.clustProb.shape

  currPerm = np.argsort(dpmObj.thetas[:,2])
  clustProbPermed = dpmObj.clustProb[:, currPerm]
  currCorrectAssignMean = np.mean([clustProbPermed[b, trueParams['clustAssignB'][b]] for b in range(nrBiomk)])

  nrIt = 1000

  for i in range(nrIt):
    # print('permutation', perm)
    newPerm = perturbPerm(currPerm)
    clustProbPermed = dpmObj.clustProb[:, newPerm]
    newCorrectAssign = np.mean([clustProbPermed[b,trueParams['clustAssignB'][b]] for b in range(nrBiomk)])

    if newCorrectAssign > currCorrectAssignMean:
      currPerm = newPerm
      currCorrectAssignMean = newCorrectAssign

      print('found better perm', newCorrectAssign, newPerm)

  # also find std
  clustProbPermed = dpmObj.clustProb[:, currPerm]
  currCorrectAssignStd = np.std([clustProbPermed[b, trueParams['clustAssignB'][b]] for b in range(nrBiomk)])

  return np.array(currPerm), currCorrectAssignMean, currCorrectAssignStd

def compareWithTrueParams(dpmObj, resStruct):

  trueParams = dpmObj.params['trueParams']
  nrBiomk, nrClust = dpmObj.clustProb.shape

  inferredPerm, voxelCorrectAssignMean, voxelCorrectAssignStd = inferPermGreedySearch(dpmObj, trueParams)
  inferredPermInv = np.argsort(inferredPerm)

  if nrClust <= 4:
    inferredPerm2, correctAssignVal2 = inferPermBrute(dpmObj, trueParams)
    print('inferredPerm', inferredPerm)
    print('inferredPerm2', inferredPerm2)
    assert ((inferredPerm - inferredPerm2) == 0).all()
    assert voxelCorrectAssignMean == correctAssignVal2
    # print(adsa)

  # print('inferredPerm', inferredPerm)
  # print('avgCorrectProb', avgCorrectProb[inferredPermInd])

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
  plotterObj = PlotterVDPM.PlotterVDPMSynth()
  # fig = plotterObj.plotSubShiftsTrue(dpmObj.subShifts, trueParams['subShiftsLong'], dpsLong, dpsLongTrue,
  #                         plotTrajParams, replaceFigMode=False, fontsize = 25)
  # fig.savefig('%s/synShiftsRes_%s.png' % (dpmObj.outFolder,
  #                                        dpmObj.params['plotTrajParams']['outFolder'].split('/')[-1]), dpi=100)

  # print(adsads)

  shiftL1MAE = np.mean(np.abs(subShiftsNaive - trueParams['subShiftsLong']), axis=0)

  # print('shiftL1mean', shiftL1mean)
  # print('shiftL1MAE', shiftL1MAE)
  # print('trueParams[subShiftsLong][:10]', trueParams['subShiftsLong'][:10])
  # print('dpmObj.subShifts[:10]', dpmObj.subShifts[:10])
  # print('subShiftsNaive[:10]', subShiftsNaive[:10])
  #
  # print('resStruct[dpsCross][4*4:5*4]', resStruct['dpsCross'][4*4:5*4])

  clustProbPermed = dpmObj.clustProb[:, inferredPerm]
  thetasPermed = dpmObj.thetas[inferredPerm, :]

  ssdThEst = np.zeros(nrClust,float)
  ssdThTrue = np.zeros(nrClust, float)
  ssdThDerivEst = np.zeros((nrClust,thetasPermed.shape[1]), float)
  ssdThDerivTrue = np.zeros((nrClust,thetasPermed.shape[1]), float)

  # see if the lik of the true thetas is higher than that of the estimated params
  # for k in range(nrClust):
  #   print(thetasPermed.shape)
  #   print(clustProbPermed.shape)
  #   print(resStruct['crossData'].shape)
  #   print(resStruct['dpsCross'].shape)
  #   ssdThEst[k] = dpmObj.objFunTheta(thetasPermed[k,:], resStruct['crossData'],
  #                                    resStruct['dpsCross'], clustProbPermed[:,k])[1]
  #   ssdThTrue[k] = dpmObj.objFunTheta(trueParams['thetas'][k,:], resStruct['crossData'],
  #                                     resStruct['dpsCross'], clustProbPermed[:,k])[1]
  #
  #   ssdThDerivEst[k] = dpmObj.objFunThetaDeriv(thetasPermed[k, :], resStruct['crossData'],
  #                                    resStruct['dpsCross'], clustProbPermed[:, k])
  #   ssdThDerivTrue[k] = dpmObj.objFunThetaDeriv(trueParams['thetas'][k, :], resStruct['crossData'],
  #                                     resStruct['dpsCross'], clustProbPermed[:, k])



  dpsCrossTrue = resStruct['crossAgeAtScan'] * trueParams['subShiftsCross'][:, 0] + \
                 trueParams['subShiftsCross'][:, 1]  # disease progression score
  clustProbTrue = makeClustProbFromArray(trueParams['clustAssignB'])
  clustProbTrueBCColNorm = clustProbTrue / np.sum(clustProbTrue, 0)[None, :]

  # fig = plotterObj.plotTrajWeightedDataMean(resStruct['crossData'], resStruct['crossDiag'], dpsCrossTrue,
  #    trueParams['thetas'][inferredPermInv,:], trueParams['variances'][inferredPermInv],
  #    clustProbTrueBCColNorm[:,inferredPermInv], dpmObj.params['plotTrajParams'], dpmObj.params['trajFunc'])
  # fig.savefig('%s/loopMeanTrue%d%d1.png' % (self.outFolder, outerIt, innerIt), dpi=100)

  # print('resStruct[crossData]',resStruct['crossData'].shape)

  # trueThetasPerturbedClustPerm = [trueParams['thetasPerturbedClust'][i] for i in  inferredPermInv]
  # fig = plotterObj.plotTrajWeightedDataMeanTrueParams(
  #   resStruct['crossData'],
  #   resStruct['crossDiag'],
  #   dpsCrossTrue,
  #   dpmObj.thetas,
  #   trueParams['variances'][inferredPermInv],
  #   clustProbTrueBCColNorm[:, inferredPermInv],
  #   dpmObj.params['plotTrajParams'],
  #   dpmObj.params['trajFunc'],
  #   trueParams['thetas'][inferredPermInv, :],
  #   showConfInt=False,
  #   trueThetasPerturbedClust=trueThetasPerturbedClustPerm, replaceFigMode=False,
  #   colorTitle=False, adjustBottomHeight=0.25, fontsize = 16)
  # fig.savefig('%s/synThetaRes_%s.png' % (dpmObj.outFolder, dpmObj.outFolder.split('/')[-1]), dpi=100)

  # the algo minimises the objFunc, so the likThTrue should be lower than likThEst
  # print('ssdThEst', ssdThEst)
  # print('ssdThTrue', ssdThTrue)
  # print('ssdThDerivEst', ssdThDerivEst)
  # print('ssdThDerivTrue', ssdThDerivTrue)

  thetaL1MAE = np.mean(np.abs(dpmObj.thetas - trueParams['thetas'][inferredPermInv, :]), axis=(0,1))

  # import pdb
  # pdb.set_trace()

  resComp = {}
  resComp['voxelCorrectAssignMean'] = voxelCorrectAssignMean
  resComp['voxelCorrectAssignStd'] = voxelCorrectAssignStd
  resComp['shiftL1MAE'] = shiftL1MAE
  resComp['thetaL1MAE'] = thetaL1MAE

  print(resComp)
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



