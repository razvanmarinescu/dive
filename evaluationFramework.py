from sklearn.metrics import *
from sklearn.model_selection import *
import math
import pickle

import os.path
from aux import *
import sys
from matplotlib import pyplot as pl
import gc

sys.path.append(os.path.abspath("../voxelwiseDPM/"))
sys.path.append(os.path.abspath("../jointModellingDisease/"))
sys.path.append(os.path.abspath("../diffEqModel/"))


# Differential equation models
from DEM import *
from DisProgBuilder import *
from aligners import *
from diffEqModel import *
from plotFunc import *

# voxelwise models
import voxelDPM
import VDPMLinear
import VDPMMean
import VDPM_MRF
import VDPMSimplified
import VDPMNan
import PlotterVDPM
import VDPMNanMasks
import VDPMNanNonMean


# joint disease models
import JointModel
import MarcoModelWrapper
import Plotter
import IncompleteModel

from importlib.machinery import SourceFileLoader
from sklearn import linear_model
plotFuncDiffEq = SourceFileLoader("*",
  os.path.abspath("../diffEqModel/plotFunc.py")).load_module()

def runModels(params, expName, modelToRun, runAllExpFunc):
  modelNames = []
  res = []

  if np.any(modelToRun == 0) or np.any(modelToRun == 1):
    fittingFunc = fitGaussianProc
    aligner = AlignerBaseVisit()  # aligns to baseline visit
    dpmBuilder = DEMBuilder(fittingFunc, aligner)  # disease progression model builder
    modelName = 'DEMStd'
    expNameCurrModel = '%s_%s' % (expName, modelName)
    params['currModel'] = 1
    res += [runAllExpFunc(params, expNameCurrModel, dpmBuilder)]
    modelNames += [modelName]

  # if np.any(modelToRun == 0) or np.any(modelToRun == 2):
  #   fittingFunc = fitGaussianProc
  #   aligner = AlignerEM()  # aligns using Expectation-Maximisation
  #   dpmBuilder = DEMBuilder(fittingFunc, aligner)  # disease progression model builder
  #   modelName = 'DEM-EM'
  #   expNameCurrModel = '%s_%s' % (expName, modelName)
  #   params['currModel'] = 2
  #   res += [runAllExpFunc(params, expNameCurrModel, dpmBuilder)]
  #   modelNames += [modelName]

  if np.any(modelToRun == 0) or np.any(modelToRun == 3):
    fittingFunc = fitGaussianProc
    aligner = AlignerLogLOpt()  # aligns using direct optimisation of incomplete logL (marginal over stages)
    dpmBuilder = DEMBuilder(fittingFunc, aligner)  # disease progression model builder
    modelName = 'DEMLogLOpt'
    expNameCurrModel = '%s_%s' % (expName, modelName)
    params['currModel'] = 3
    res += [runAllExpFunc(params, expNameCurrModel, dpmBuilder)]
    modelNames += [modelName]

  if np.any(modelToRun == 0) or np.any(modelToRun == 4):
    # Voxelwise dpm with sigmoid trajectories and dynamic ROIs
    dpmBuilder = voxelDPM.VoxelDPMBuilder(params['cluster'])  # disease progression model builder
    modelName = 'VWDPMStd'
    expNameCurrModel = '%s_%s' % (expName, modelName)
    params['currModel'] = 4
    res += [runAllExpFunc(params, expNameCurrModel, dpmBuilder)]
    modelNames += [modelName]

  if np.any(modelToRun == 0) or np.any(modelToRun == 5):
    # as above but linear trajectories
    dpmBuilder = VDPMLinear.VDPMLinearBuilder(params['cluster'])  # disease progression model builder
    modelName = 'VWDPMLinear'
    expNameCurrModel = '%s_%s' % (expName, modelName)
    params['currModel'] = 5
    res += [runAllExpFunc(params, expNameCurrModel, dpmBuilder)]
    modelNames += [modelName]

  if np.any(modelToRun == 0) or np.any(modelToRun == 6):
    # Voxelwise dpm with sigmoid trajectories and static ROIs, inherits VDPMMean
    dpmBuilder = VDPMSimplified.VDPMStaticBuilder(params['cluster'])  # disease progression model builder
    modelName = 'VWDPMStatic'
    expNameCurrModel = '%s_%s' % (expName, modelName)
    params['currModel'] = 6
    res += [runAllExpFunc(params, expNameCurrModel, dpmBuilder)]
    modelNames += [modelName]

  if np.any(modelToRun == 0) or np.any(modelToRun == 7):
    # Voxelwise dpm with linear trajectories and static ROIs
    dpmBuilder = VDPMLinear.VDPMLinearStaticBuilder(params['cluster'])  # disease progression model builder
    modelName = 'VWDPMLinearStatic'
    expNameCurrModel = '%s_%s' % (expName, modelName)
    params['currModel'] = 7
    res += [runAllExpFunc(params, expNameCurrModel, dpmBuilder)]
    modelNames += [modelName]

  if np.any(modelToRun == 0) or np.any(modelToRun == 8):
    # VDPM with sigmoids that uses the mean across all voxels in the cluster for the obj func
    dpmBuilder = VDPMMean.VDPMMeanBuilder(params['cluster'])  # disease progression model builder
    modelName = 'VWDPMMean'
    expNameCurrModel = '%s_%s' % (expName, modelName)
    params['currModel'] = 8
    res += [runAllExpFunc(params, expNameCurrModel, dpmBuilder)]
    modelNames += [modelName]

  if np.any(modelToRun == 0) or np.any(modelToRun == 9):
    # VDPM_MRF mean model with MRF (Markos Random Field)
    dpmBuilder = VDPM_MRF.VDPMMrfBuilder(params['cluster'])  # disease progression model builder
    modelName = 'VDPM_MRF'
    expNameCurrModel = '%s_%s' % (expName, modelName)
    params['currModel'] = 9
    res += [runAllExpFunc(params, expNameCurrModel, dpmBuilder)]
    modelNames += [modelName]

  if np.any(modelToRun == 0) or np.any(modelToRun == 10):
    # VDPM without subject staging
    dpmBuilder = VDPMSimplified.VDPMNoDPSBuilder(params['cluster'])  # disease progression model builder
    modelName = 'VDPMNoDPS'
    expNameCurrModel = '%s_%s' % (expName, modelName)
    params['currModel'] = 10
    res += [runAllExpFunc(params, expNameCurrModel, dpmBuilder)]
    modelNames += [modelName]

  if np.any(modelToRun == 0) or np.any(modelToRun == 11):
    # VDPM that works with missing data (NaNs)
    dpmBuilder = VDPMNan.VDPMNanBuilder(params['cluster'])  # disease progression model builder
    dpmBuilder.setPlotter(PlotterVDPM.PlotterVDPMScalar())
    modelName = 'VDPMNan'
    expNameCurrModel = '%s_%s' % (expName, modelName)
    params['currModel'] = 11
    res += [runAllExpFunc(params, expNameCurrModel, dpmBuilder)]
    modelNames += [modelName]

  if np.any(modelToRun == 0) or np.any(modelToRun == 12):
    # VDPM for NaNs that uses marked arrays
    dpmBuilder = VDPMNanMasks.VDPMNanMasksBuilder(params['cluster'])  # disease progression model builder
    dpmBuilder.setPlotter(PlotterVDPM.PlotterVDPMScalar())
    modelName = 'VDPMNanMasks'
    expNameCurrModel = '%s_%s' % (expName, modelName)
    params['currModel'] = 12
    res += [runAllExpFunc(params, expNameCurrModel, dpmBuilder)]
    modelNames += [modelName]

  if np.any(modelToRun == 0) or np.any(modelToRun == 13):
    # VDPM for NaNs that doesn't use the fast implementation, which created some problems
    # as it biased the SSD calculation, since when there was missing data corresponding to a high
    # clustering probability, the error was dominated by the other (present) biomkarker data that
    # had correspondingly low clustering probabilities for that particular cluster
    dpmBuilder = VDPMNanNonMean.VDPMNanNonMeanBuilder(params['cluster'])  # disease progression model builder
    dpmBuilder.setPlotter(PlotterVDPM.PlotterVDPMScalar())
    modelName = 'VDPMNanNonMean'
    expNameCurrModel = '%s_%s' % (expName, modelName)
    params['currModel'] = 13
    res += [runAllExpFunc(params, expNameCurrModel, dpmBuilder)]
    modelNames += [modelName]

  if np.any(modelToRun == 0) or np.any(modelToRun == 14):
    # JMD - Joint Model of Diseases
    dpmBuilder = JointModel.JMDBuilder(params['plotTrajParams'])
    modelName = 'JMD'
    expNameCurrModel = '%s_%s' % (expName, modelName)
    params['currModel'] = 14
    res += [runAllExpFunc(params, expNameCurrModel, dpmBuilder)]
    modelNames += [modelName]

  if np.any(modelToRun == 0) or np.any(modelToRun == 15):
    # Marco's Model
    dpmBuilder = MarcoModelWrapper.MarcoModelBuilder(params['plotTrajParams'])
    modelName = 'MarcoModel'
    expNameCurrModel = '%s_%s' % (expName, modelName)
    params['currModel'] = 15
    res += [runAllExpFunc(params, expNameCurrModel, dpmBuilder)]
    modelNames += [modelName]

  if np.any(modelToRun == 0) or np.any(modelToRun == 16):
    # Incomplete JMD - Joint Model of Diseases
    filePathUnitModels = params['filePathUnitModels']
    gpModels = pickle.load(open(filePathUnitModels, 'rb'))
    dpmBuilder = IncompleteModel.IncompleteBuilder(params['plotTrajParams'], gpModels)
    modelName = 'IJDM'
    expNameCurrModel = '%s_%s' % (expName, modelName)
    params['currModel'] = 16
    res += [runAllExpFunc(params, expNameCurrModel, dpmBuilder)]
    modelNames += [modelName]

  return modelNames, res

def runStdDPM(params, expNameCurrModel, dpmBuilder, runPart):
  dataIndices = np.logical_not(np.in1d(params['diag'], params['excludeXvalidID']))
  print(np.sum(np.logical_not(dataIndices)))
  print('excludeID', params['excludeXvalidID'])
  print(params['diag'].shape)
  dpmObj = dpmBuilder.generate(dataIndices, expNameCurrModel, params)
  res = None
  if runPart[0] == 'R':
    res = dpmObj.runStd(params['runPartStd'])

  if runPart[1] == 'R':
    dpmObj.plotTrajectories(res)

  if runPart[2] == 'R':
    # dataIndicesNN = np.logical_and(dataIndices, np.sum(np.isnan(params['data']),1) == 0)
    (maxLikStages, maxStagesIndex, stagingProb, stagingLik, tsStages, _) = dpmObj.stageSubjects(dataIndices)
    print(params['diag'].shape, dataIndices.shape)
    print('maxLikStages min max', np.min(maxLikStages), np.max(maxLikStages))
    fig, lgd = plotFuncDiffEq.plotStagingHist(maxLikStages, diag=params['diag'][dataIndices],
                    plotTrajParams=params['plotTrajParams'], expNameCurrModel=expNameCurrModel)
    stagingHistFigName = '%s/stagingHist.png' % dpmObj.outFolder
    fig.savefig(stagingHistFigName, bbox_extra_artists=(lgd,), bbox_inches='tight')


  return dpmObj, res

def crossValidAndCorrCog(dpmBuilder, expName, params):
  statsFile = 'resfiles/cogCorr/%s/stats.npz' % expName
  nrProcesses = params['nrProcesses']
  runIndex = params['runIndex']
  nrFolds = 10
  procResFile = ['resfiles/cogCorr/%s/procRes_n%d_p%d.npz' % (expName, nrProcesses, p)
                 for p in range(1,nrProcesses+1)]
  nrCogTests = params['cogTests'].shape[1]
  nrBiomk = params['data'].shape[1]
  nrClust = params['nrClust']
  seed = 1

  makeScanTimeptsStartFromOne(params)

  if params['masterProcess']:
    if params['runPartCogCorr'][0] == 'R' or params['runPartCogCorr'][0] == 'L':
      savedData0 = pickle.load(open(procResFile[0], 'rb'))
      nrParamsTheta = savedData0['paramsCurrProc'][0][0].shape[1]
      nrSubjLong =  savedData0['paramsCurrProc'][0][2].shape[0]

      thetasAllFolds = np.zeros((nrFolds, nrClust, nrParamsTheta), float)
      variancesAllFolds = np.zeros((nrFolds, nrClust, nrParamsTheta), float)
      subShiftsAllFolds = np.zeros((nrFolds, nrClust, nrSubjLong), float)

      statsAllFolds = np.nan * np.ones((nrFolds,2*nrCogTests), float)
      clustProbAllFolds = np.zeros((nrFolds, nrBiomk, nrClust), float)
      testPredPredFPB = [0 for x in range(nrFolds)]
      testPredDataFPB = [0 for x in range(nrFolds)]

      pValAllFolds = np.nan * np.ones((nrFolds,2*nrCogTests), float)

      longMaxLikStagesIndep = []
      longMaxLikStages = []
      longDiagTestList = []
      longAgeAtScanTestList = []
      cogTestsTestList = [0 for i in range(nrFolds)]
      maxLikStagesTestList = [0 for i in range(nrFolds)]
      predStats = np.nan * np.ones(nrFolds, float)

      for p in range(nrProcesses):
        if os.path.isfile(procResFile[p]):
          savedData = pickle.load(open(procResFile[p], 'rb'))
          statsAllFolds[savedData['foldInstances']] = savedData['statsCurrProc']
          pValAllFolds[savedData['foldInstances']] = savedData['pValCurrProc']
          paramsList = savedData['paramsCurrProc']
          for f in range(len(savedData['foldInstances'])):
            foldIndex = savedData['foldInstances'][f]
            # paramsList = [thetas, variances, subShifts, clustProb]
            thetasAllFolds[foldIndex,:,:] = paramsList[f][0]
            clustProbAllFolds[foldIndex,:,:] = paramsList[f][3]
            if 'testPredPredPB' in savedData.keys():
              testPredPredFPB[foldIndex] = savedData['testPredPredPB'][f]
              testPredDataFPB[foldIndex] = savedData['testPredData'][f]

            longMaxLikStages += savedData['longMaxLikStagesTest'][f]
            longDiagTestList += [savedData['longDiagTest'][f]]
            longAgeAtScanTestList += [savedData['longAgeAtScanTest'][f]]

            cogTestsTestList[foldIndex] = savedData['cogTestsTest'][f]
            maxLikStagesTestList[foldIndex] = savedData['maxLikStagesTest'][f]
            predStats[foldIndex] = np.array(savedData['predStats'])

            print(' fold %d ----------' % foldIndex)
            print('longMaxLikStages', longMaxLikStages)
            print('cogTestsTest', savedData['cogTestsTest'])
            print('cogTestsTestList', cogTestsTestList[foldIndex])
            print('maxLikStagesTestList', maxLikStagesTestList[foldIndex])
            print('predStats', predStats[foldIndex])

      statsAll = {#'mean': np.mean(statsAllFolds, axis=0),
                  #'std': np.std(statsAllFolds, axis=0),

                  'statsAllFolds':statsAllFolds, 'predStats': predStats,
                  'testPredPredFPB':testPredPredFPB, 'testPredDataFPB':clustProbAllFolds}


      plotTrajParams = params['plotTrajParams']
      plotTrajParams['outFolder'] = 'resfiles/cogCorr/%s' % expName

      ################# make scatter plots of cognitive tests with DPS ##################

      cogTestsTest = cogTestsTestList[0]
      maxLikStagesTest = maxLikStagesTestList[0]
      longDiagTest = longDiagTestList[0]
      longAgeAtScanTest = longAgeAtScanTestList[0]
      # print('cogTestsTestList', cogTestsTestList)
      # print('maxLikStagesTestList', maxLikStagesTestList)
      for f in range(1,nrFolds):
        # print('cogTestsTestList[f].shape', cogTestsTestList[f].shape)
        # print('maxLikStagesTestList[f].shape', maxLikStagesTestList[f].shape)
        cogTestsTest = np.append(cogTestsTest, cogTestsTestList[f], axis=0)
        maxLikStagesTest = np.append(maxLikStagesTest, maxLikStagesTestList[f], axis=0)
        longDiagTest = np.append(longDiagTest, longDiagTestList[f], axis=0)
        longAgeAtScanTest = np.append(longAgeAtScanTest, longAgeAtScanTestList[f], axis=0)

      print('longDiagTest', len(longDiagTest))
      print('longAgeAtScanTest', len(longAgeAtScanTest))
      print('longMaxLikStages', len(longMaxLikStages))
      assert len(longDiagTest) == len(longMaxLikStages)
      assert len(longAgeAtScanTest) == len(longMaxLikStages)
      diagNrs = np.unique(longDiagTest)
      nrUnqDiag = diagNrs.shape[0]

      testLabels = ['CDRSOB', 'ADAS13', 'MMSE', 'RAVLT']
      nrTests = len(testLabels)
      inversionFactors = np.array([1,1,-1,-1])

      cogTestsTest = cogTestsTest * inversionFactors[None,:]

      corrAllTestData = [0 for x in range(nrTests)]
      pValAllTestData = [0 for x in range(nrTests)]

      for t in range(nrTests):
        # for d in range(nrUnqDiag):
        #   pl.scatter(maxLikStagesTest[longDiag == diagNrs[d]], cogTestsTest[longDiag == diagNrs[d],t])
        fig = pl.figure()
        notNanMask = np.logical_not(np.logical_or(np.isnan(maxLikStagesTest), np.isnan(cogTestsTest[:,t])))

        pl.scatter(maxLikStagesTest, cogTestsTest[:, t])
        regr = linear_model.LinearRegression()
        regr.fit(maxLikStagesTest[notNanMask].reshape(-1,1), cogTestsTest[notNanMask, t].reshape(-1,1))
        predictedVals = regr.predict(maxLikStagesTest[notNanMask].reshape(-1,1)).reshape(-1)

        corrAllTestData[t], pValAllTestData[t] = scipy.stats.pearsonr(
          maxLikStagesTest[notNanMask].reshape(-1, 1), cogTestsTest[notNanMask, t].reshape(-1, 1))

        pl.plot(maxLikStagesTest[notNanMask].reshape(-1),predictedVals, color='k', linewidth=3)

        fs = 18
        pl.xlabel('Disease Progression Score (DPS)',fontsize=fs)
        pl.ylabel(testLabels[t],fontsize=fs)
        pl.xticks(fontsize=fs)
        pl.yticks(fontsize=fs)
        pl.gcf().subplots_adjust(bottom=0.15,left=0.15)

        fig.show()
        fig.savefig('%s/stagingCogTestsScatterPlot_%s_%s.png' % (plotTrajParams['outFolder'],
          expName, testLabels[t]), dpi=100)

      statsAll['corrAllTestData'] = corrAllTestData
      statsAll['pValAllTestData'] = pValAllTestData
      print('corrAllTestData', corrAllTestData)
      print('pValAllTestData', pValAllTestData)

      # print(adsas)

        # print(adsa)

      ######### find average dice score of clusters across all pairs of folds ############

      # first sort the clusters according to slope
      clustProbAllFoldsSorted = clustProbAllFolds
      for f in range(nrFolds):
        slopes = (thetasAllFolds[f,:, 0] * thetasAllFolds[f,:, 1]) / 4
        slopesSortedInd = np.argsort(slopes) # clust 0 -> clust 1 -> clust 2

        clustProbAllFoldsSorted[f,:,:] = clustProbAllFolds[f,:,slopesSortedInd].T

      maxLikClustFB = np.argmax(clustProbAllFoldsSorted, axis=2)
      dice = np.zeros((nrClust, nrFolds, nrFolds), float)
      for c in range(nrClust):
        for f1 in range(nrFolds):
          for f2 in range(nrFolds):
            # find union - nr of vertices that match
            union = np.sum(np.logical_and(maxLikClustFB[f1,:] == c, maxLikClustFB[f2,:] == c))
            # divide by nr_vertices in image

            dice[c,f1,f2] = 2*union / (np.sum(maxLikClustFB[f1,:] == c) +
                            np.sum(maxLikClustFB[f2,:] == c))

        print(dice[0,:,:])

      avgDiceC = np.mean(dice, axis=(1,2))
      for c in range(nrClust):
        print('avg dice for c %d' % c, avgDiceC[c])

      print('avgDiceOverall', np.mean(avgDiceC))
      # print(adsa)


      ################## plot staging consistency ########################

      fig = plotStagingConsist(plotTrajParams, longAgeAtScanTest,
                               longMaxLikStages, longDiagTest)
      fileName = '%s/stagingConsist_%s.png' % (plotTrajParams['outFolder'],expName)
      print('Saving file: %s' % fileName)
      fig.savefig(fileName, dpi=100)


      # save clustProb to file, run blender on them
      # nnDataFile = 'resfiles/%sNearNeigh.npz' % params['datasetFull']
      # nnDataStruct = pickle.load(open(nnDataFile, 'rb'))
      # plotTrajParams['nearestNeighbours'] = nnDataStruct['nearestNeighbours']

      # statsStruct = dict(statsAllFolds=statsAllFolds, clustProbOBC=clustProbAllFolds,
      #                    plotTrajParams=plotTrajParams, thetas=thetasAllFolds)
      # pickle.dump(statsStruct, open(statsFile, 'wb'), protocol = pickle.HIGHEST_PROTOCOL)
      # makeSnapshotBlender(statsFile, plotTrajParams)

      pass
    else:
      statsAll = None
  else:
    if params['runPartCogCorr'][0] == 'R':

      # cogTestNonNanInd = np.logical_not(np.isnan(params['cogTest1']))
      excludeDiagIdInd = np.logical_not(np.in1d(params['diag'], params['excludeXvalidID']))
      # filterAllInd = np.logical_and(cogTestNonNanInd, excludeDiagIdInd)
      if params['datasetFull'].startswith('drc'):
        # need to clone as this is used twice for PCA and tAD.
        filteredParams = filterDDSPAIndices(params, excludeDiagIdInd)
      else:
        filteredParams = filterDDSPAIndicesShallow(params, excludeDiagIdInd)
      filteredParams['cogTests'] = filteredParams['cogTests'][excludeDiagIdInd,:]

      filteredParams['plotTrajParams']['TrajSamplesFontSize'] = 17
      filteredParams['plotTrajParams']['TrajSamplesAdjBottomHeight'] = 0.22
      filteredParams['plotTrajParams']['trajSamplesPlotLegend'] = False

      # print(len(filteredParams['acqDate']), excludeDiagIdInd.shape[0])
      # filteredParams['acqDate'] = [filteredParams['acqDate'][i] for i in
      #                              range(excludeDiagIdInd.shape[0]) if excludeDiagIdInd[i]]

      assert(filteredParams['cogTests'].shape[0] == filteredParams['data'].shape[0])

      foldInstances = allocateRunIndicesToProcess(nrFolds, nrProcesses, runIndex)
      nrFoldsCurrProc = len(foldInstances)

      (uniquePatCode, uniqueIndices) = np.unique(filteredParams['partCode'], return_index=True)
      nrUniquePat = uniquePatCode.shape[0]
      nrPat = filteredParams['diag'].shape[0]
      uniqueDiag = filteredParams['diag'][uniqueIndices]
      # trainIndices = np.zeros((nrFolds, nrPat), bool)
      # testIndices = np.zeros((nrFolds, nrPat), bool)
      skf = StratifiedKFold(n_splits = nrFolds, shuffle= True, random_state = seed)
      foldIndGen = list(skf.split(uniquePatCode,np.zeros(nrUniquePat), uniqueDiag))

      dpmObj = [0 for x in range(nrFoldsCurrProc)]
      # train the progression model on the training Data
      maxLikStagesLong = []
      stagingProbLong = []

      # coffCoef
      statsCurrProc = np.zeros((nrFoldsCurrProc, 2*nrCogTests), float) # both pearson and spearman
      pValCurrProc = np.zeros((nrFoldsCurrProc, 2*nrCogTests), float)  # both pearson and spearman

      paramsCurrProc = [0 for x in range(nrFoldsCurrProc)] # for plotting them
      longMaxLikStagesTest = [0 for x in range(nrFoldsCurrProc)]
      longDiagTestList = [0 for x in range(nrFoldsCurrProc)]
      longAgeAtScanTestList = [0 for x in range(nrFoldsCurrProc)]
      cogTestsTest = [0 for x in range(nrFoldsCurrProc)]
      predStats = [0 for x in range(nrFoldsCurrProc)]
      maxLikStagesTest = [0 for x in range(nrFoldsCurrProc)]
      testPredPredPB = [0 for x in range(nrFoldsCurrProc)]
      testPredData = [0 for x in range(nrFoldsCurrProc)]

      for fld in range(nrFoldsCurrProc):
        foldIndex = foldInstances[fld]
        print(len(foldIndGen))
        (trainIndicesUnq, testIndicesUnq) = foldIndGen[foldIndex]
        trainIndices = np.in1d(filteredParams['partCode'], uniquePatCode[trainIndicesUnq])
        testIndices = np.in1d(filteredParams['partCode'], uniquePatCode[testIndicesUnq])

        print(testIndices.shape, filteredParams['scanTimepts'].shape)
        print("foldIndex %d" % foldIndex)

        expNameCurrFold = 'cogCorr/%s/f%d' % (expName, foldIndex)
        dpmObj[fld] = dpmBuilder.generate(trainIndices, expNameCurrFold, filteredParams)
        print(dpmObj)

        dpmRes = dpmObj[fld].run(params['runPartCogCorrMain'])
        
        # dpmObj[fld].plotTrajSummary(dpmRes)
        # ( _, _, stagingProbTrain, _,_) = dpmObj[fld].stageSubjects(trainIndices)

        print('data.shape', filteredParams['data'].shape)
        print('testIndices.shape', testIndices.shape)
        print('filteredParams[scanTimepts].shape', filteredParams['scanTimepts'].shape)
        print('trainIndices.shape', trainIndices.shape)
        print('cogTests', filteredParams['cogTests'][testIndices, :])
        # print(adsa)
        assert(filteredParams['data'].shape[0] == testIndices.shape[0])

        (maxLikStagesTest[fld], _, _, _,longMaxLikStagesTest[fld], otherParams) = \
          dpmObj[fld].stageSubjects(testIndices)

        longDiagTestList[fld] = otherParams['longDiag']
        longAgeAtScanTestList[fld] = otherParams['longAgeAtScan']

        # correlate with chosen cognitive tests
        cogTestsTest[fld] = filteredParams['cogTests'][testIndices, :]
        statsCurrProc[fld,:], pValCurrProc[fld,:] = corrCog(maxLikStagesTest[fld],
          cogTestsTest[fld])


        # predict future biomarker values (for voxelwise predict each cort. thick. vertex)
        testInputInd = np.logical_and(testIndices, filteredParams['scanTimepts'] <= 2)
        testPredInd = np.logical_and(testIndices, filteredParams['scanTimepts'] > 2)
        # print(adsa)

        predStats[fld], testPredPredPB[fld], testPredData[fld] = dpmObj[fld].calcPredScores(testInputInd, testPredInd,
          filteredParams)

        paramsCurrProc[fld] = dpmObj[fld].getFittedParams()
        print('statsCurrProc[fld]', statsCurrProc[fld])
        print('pValCurrProc[fld]', pValCurrProc[fld])
        print('predStats[fld]', predStats[fld])

      savedData = dict(statsCurrProc=statsCurrProc,
                       pValCurrProc=pValCurrProc, cogTestsTest=cogTestsTest,
                       maxLikStagesTest=maxLikStagesTest,
                       paramsCurrProc=paramsCurrProc,
                       foldInstances=foldInstances,
                       longMaxLikStagesTest=longMaxLikStagesTest,
                       longDiagTest=longDiagTestList,
                       predStats=predStats,
                       longAgeAtScanTest=longAgeAtScanTestList,
                       # testPredPredPB=testPredPredPB,
                       # testPredData=testPredData
                       foldIndGen=foldIndGen
                       )
      pickle.dump(savedData, open(procResFile[runIndex-1], 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    statsAll = None

  return statsAll

def makeScanTimeptsStartFromOne(params):

  unqPartCodes = np.unique(params['partCode'])

  for p in range(unqPartCodes.shape[0]):
    timeptsCurrPart = params['scanTimepts'][params['partCode'] == unqPartCodes[p]]
    timeptsConsec = 1+ np.argsort(np.argsort(timeptsCurrPart))
    # print('timeptsCurrPart', timeptsCurrPart)
    # print('timeptsConsec', timeptsConsec)
    # print(params['scanTimepts'][params['partCode'] == unqPartCodes[p]])
    params['scanTimepts'][params['partCode'] == unqPartCodes[p]] = timeptsConsec
    # print(params['scanTimepts'][params['partCode'] == unqPartCodes[p]])
    # print('----')
  # print(adsa)

def cvNonOverlapFolds(dpmBuilder, expName, params):
  outFolder = 'resfiles/cvNonOverlap/%s/' % expName
  statsFile = '%s/stats.npz'  % outFolder
  nrProcesses = params['nrProcesses']
  runIndex = params['runIndex']
  nrFolds = 3
  nrIter = 4
  procResFile = ['%s/procRes_n%d_p%d.npz' % (outFolder, nrProcesses, p)
    for p in range(1, nrProcesses + 1)]
  nrCogTests = params['cogTests'].shape[1]
  nrBiomk = params['data'].shape[1]
  nrClust = params['nrClust']

  makeScanTimeptsStartFromOne(params)

  if params['runPartCVNonOverlap'][0] == 'R':

    # cogTestNonNanInd = np.logical_not(np.isnan(params['cogTest1']))
    excludeDiagIdInd = np.logical_not(np.in1d(params['diag'], params['excludeXvalidID']))
    # filterAllInd = np.logical_and(cogTestNonNanInd, excludeDiagIdInd)
    if params['datasetFull'].startswith('drc'):
      # need to clone as this is used twice for PCA and tAD.
      filteredParams = filterDDSPAIndices(params, excludeDiagIdInd)
    else:
      filteredParams = filterDDSPAIndicesShallow(params, excludeDiagIdInd)

    filteredParams['cogTests'] = filteredParams['cogTests'][excludeDiagIdInd, :]

    # print(len(filteredParams['acqDate']), excludeDiagIdInd.shape[0])
    # filteredParams['acqDate'] = [filteredParams['acqDate'][i] for i in
    #                              range(excludeDiagIdInd.shape[0]) if excludeDiagIdInd[i]]

    assert (filteredParams['cogTests'].shape[0] == filteredParams['data'].shape[0])

    foldInstances = allocateRunIndicesToProcess(nrFolds, nrProcesses, runIndex)
    nrFoldsCurrProc = len(foldInstances)

    (uniquePatCode, uniqueIndices) = np.unique(filteredParams['partCode'], return_index=True)
    nrUniquePat = uniquePatCode.shape[0]
    nrPat = filteredParams['diag'].shape[0]
    uniqueDiag = filteredParams['diag'][uniqueIndices]
    # trainIndices = np.zeros((nrFolds, nrPat), bool)
    # testIndices = np.zeros((nrFolds, nrPat), bool)

    for i in range(nrIter):
      seed = i # set different fold allocations at each iterations
      skf = StratifiedKFold(n_splits=nrFolds, shuffle=True, random_state=seed)
      foldIndGen = list(skf.split(uniquePatCode, np.zeros(nrUniquePat), uniqueDiag))

      dpmObj = [0 for x in range(nrFoldsCurrProc)]


      predStats = [0 for x in range(nrFoldsCurrProc)]
      # maxLikStagesTest = [0 for x in range(nrFoldsCurrProc)]

      for fld in range(nrFoldsCurrProc):
        foldIndex = foldInstances[fld]
        print(len(foldIndGen))
        (_, testIndicesUnq) = foldIndGen[foldIndex]
        testIndices = np.in1d(filteredParams['partCode'], uniquePatCode[testIndicesUnq])

        print("foldIndex %d" % foldIndex)

        expNameCurrFold = 'cvNonOverlap/%s/i%d_f%d' % (expName, i, foldIndex)
        dpmObj[fld] = dpmBuilder.generate(testIndices, expNameCurrFold, filteredParams)
        dpmObj[fld].run(params['runPartCVNonOverlapMain'])


def corrCog(maxLikStagesTest, cogTestsTest):
  # correlate with chosen cognitive tests
  nrCogTests = cogTestsTest.shape[1]
  assert (cogTestsTest.shape[0] == maxLikStagesTest.shape[0])
  cogTestNonNanInd = np.logical_not(np.sum(np.isnan(cogTestsTest), axis=1) > 0)
  cogTestsNotNanValues = cogTestsTest[cogTestNonNanInd, :]


  statsCurrProc = np.zeros(2*nrCogTests, float)
  pValCurrProc = np.zeros(2*nrCogTests, float)

  for t in range(nrCogTests):
    rho, pVal = scipy.stats.pearsonr(cogTestsNotNanValues[:, t], maxLikStagesTest[cogTestNonNanInd])
    rhoSpear, pValSpear = scipy.stats.spearmanr(cogTestsNotNanValues[:, t], maxLikStagesTest[cogTestNonNanInd])
    # print(np.sum(np.isnan(params['cogTests'])))
    print('rho, rhoSpear', rho, rhoSpear)
    # print(asdsa)
    statsCurrProc[t] = rho
    statsCurrProc[t + nrCogTests] = rhoSpear
    pValCurrProc[t] = pVal
    pValCurrProc[t + nrCogTests] = pValSpear

  return statsCurrProc, pValCurrProc

def evalDiffDiagDRC(dpmBuilder, expName, params):
  statsFile = 'matfiles/diffDiag/%s/stats.npz' % expName
  nrProcesses = params['nrProcesses']
  runIndex = params['runIndex']
  nrFolds = 10
  procResFile = ['matfiles/diffDiag/%s/procRes_n%d_p%d.npz' % (expName, nrProcesses, p)
                 for p in range(1,nrProcesses+1)]

  if params['masterProcess']:
    if params['runPartDiffDiag'][2] == 'R' or params['runPartDiffDiag'][2] == 'L':
      diagStatsAll = np.zeros((nrFolds,6))
      for p in range(nrProcesses):
        if os.path.isfile(procResFile[p]):
          savedData = pickle.load(open(procResFile[p], 'rb'))
          diagStatsAll[savedData['foldInstances'],:] = savedData['diagStatsCurrProc']
        else:
          raise IOError("file %s found" % procResFile[p])

      diagStats = {'mean': np.mean(diagStatsAll, axis=0), 'std': np.std(diagStatsAll, axis=0)}
      pass
    else:
      diagStats = None
  else:
    if params['runPartDiffDiag'][2] == 'R':
      foldInstances = allocateRunIndicesToProcess(nrFolds, nrProcesses, runIndex)
      nrFoldsCurrProc = len(foldInstances)

      (uniquePatCode, uniqueIndices) = np.unique(params['partCode'], return_index=True)
      nrUniquePat = uniquePatCode.shape[0]
      nrPart = params['diag'].shape[0]
      nrUniquePart = uniquePatCode.shape[0]
      uniqueDiag = params['diag'][uniqueIndices]
      uniquePartCodeCTL = uniquePatCode[uniqueDiag == CTL]
      uniqueDiagPcaAdIndices= np.logical_or(uniqueDiag == PCA, uniqueDiag == AD)
      uniquePartCodePcaAd = uniquePatCode[uniqueDiagPcaAdIndices]

      # trainIndices = np.zeros((nrFolds, nrPart), bool)
      # testIndices = np.zeros((nrFolds, nrPart), bool)
      seed = 1
      skf = StratifiedKFold(n_splits = nrFolds, shuffle= True, random_state = seed)
      foldIndGen = skf.split(uniquePartCodePcaAd,np.zeros(np.sum(uniqueDiagPcaAdIndices)),
                             uniqueDiag[uniqueDiagPcaAdIndices])
      foldIndGenList = [x for x in foldIndGen]

      dpmObjPCA = [0 for x in range(nrFoldsCurrProc)]
      dpmObjAD = [0 for x in range(nrFoldsCurrProc)]
      # train the progression model on the training Data
      maxLikStagesLong = []
      stagingProbLong = []

      diagStatsCurrProc = np.zeros((nrFoldsCurrProc,6)) # acc, sens, spec, balAcc, #visitsClass1, #visitsClass2, #subjClass1,
        #subjClass2
      confMatCurrProc = np.zeros((nrFoldsCurrProc,2,2))
      diagClasses = [PCA, AD]

      for fld in range(nrFoldsCurrProc):
        foldIndex = foldInstances[fld]
        (trainIndicesUnq, testIndicesUnq) = foldIndGenList[foldIndex]
        # print("testIndFold1 should be same regardless of process", foldIndGenList[0][1])

        # print(np.concatenate((uniquePartCodePcaAd[trainIndicesUnq], uniquePartCodeCTL)),'----')
        # print(uniquePartCodePcaAd[trainIndicesUnq],'-----', uniquePartCodeCTL)
        trainIndices = np.in1d(params['partCode'],
          np.concatenate((uniquePartCodePcaAd[trainIndicesUnq], uniquePartCodeCTL)))
        testIndices = np.in1d(params['partCode'], uniquePartCodePcaAd[testIndicesUnq])

        # PCA
        expNameCurrFoldPCA = 'diffDiag/%s/pca_f%d' % (expName, foldIndex)
        # params['patientID'] = PCA
        params['excludeID'] = [AD]
        params['anchorID'] = PCA
        params['excludeXvalidID'] = [AD]
        dpmObjPCA[fld] = dpmBuilder.generate(trainIndices, expNameCurrFoldPCA, params)
        dpmResPCA = dpmObjPCA[fld].run(params['runPartDiffDiag'])
        # dpmObjPCA[fld].plotTrajSummary(dpmResPCA)

        (_, _, _, stagingLikTestPCA,_, _) = dpmObjPCA[fld].stageSubjects(testIndices)

        # AD
        expNameCurrFoldAD = 'diffDiag/%s/ad_f%d' % (expName, foldIndex)
        # params['patientID'] = AD
        params['excludeID'] = [PCA]
        params['anchorID'] = AD
        params['excludeXvalidID'] = [PCA]
        dpmObjAD[fld] = dpmBuilder.generate(trainIndices, expNameCurrFoldAD, params)
        dpmResAD = dpmObjAD[fld].run(params['runPartDiffDiag'])
        # dpmObjAD[fld].plotTrajSummary(dpmResAD)

        (_, _, _, stagingLikTestAD,_, _) = dpmObjAD[fld].stageSubjects(testIndices)

        # diagnosis between PCA and AD
        trueDiag = params['diag'][testIndices]
        predDiag = PCA * np.ones(trueDiag.shape)
        predDiag[np.sum(stagingLikTestAD, axis=1) > np.sum(stagingLikTestPCA, axis=1)] = AD
        confMatCurrProc[fld,:,:] = confusion_matrix(trueDiag, predDiag)
        diagStatsCurrProc[fld, 0:4] = getSensitivitySpecificity(confMatCurrProc[fld,:,:])
        diagStatsCurrProc[fld, 4:6] = [np.sum(uniqueDiag[testIndicesUnq] == diagClasses[0]),
                                        np.sum(uniqueDiag[testIndicesUnq] == diagClasses[1])]


      savedData = dict(diagStatsCurrProc=diagStatsCurrProc,
        confMatCurrProc=confMatCurrProc,
        foldInstances=foldInstances)

      fh = open(procResFile[runIndex-1], 'wb')
      pickle.dump(savedData, fh, protocol=pickle.HIGHEST_PROTOCOL)

    diagStats = None

  return diagStats

def evalDirDiag(dpmBuilder, expName, params):
  statsFile = 'matfiles/dirDiag/%s/stats.npz' % expName
  nrProcesses = params['nrProcesses']
  runIndex = params['runIndex']
  nrFolds = 10
  procResFile = ['matfiles/dirDiag/%s/procRes_n%d_p%d.npz' % (expName, nrProcesses, p)
                 for p in range(1,nrProcesses+1)]

  if params['masterProcess']:
    if params['runPartDirDiag'][2] == 'R' or params['runPartDirDiag'][2] == 'L':
      diagStatsAll = np.zeros((nrFolds,8))
      for p in range(nrProcesses):
        if os.path.isfile(procResFile[p]):
          savedData = pickle.load(open(procResFile[p], 'rb'))
          diagStatsAll[savedData['foldInstances'],:] = savedData['diagStatsCurrProc']
        else:
          raise IOError("file %s found" % procResFile[p])

      diagStats = {'mean': np.mean(diagStatsAll, axis=0), 'std': np.std(diagStatsAll, axis=0)}
      pass
    else:
      diagStats = None
  else:
    if params['runPartDirDiag'][2] == 'R':
      filteredParams = filterDDSPA(params, params['excludeXvalidID'])
      foldInstances = allocateRunIndicesToProcess(nrFolds, nrProcesses, runIndex)
      nrFoldsCurrProc = len(foldInstances)

      (uniquePatCode, uniqueIndices) = np.unique(filteredParams['partCode'], return_index=True)
      nrUniquePat = uniquePatCode.shape[0]
      nrPat = filteredParams['diag'].shape[0]
      uniqueDiag = filteredParams['diag'][uniqueIndices]
      # trainIndices = np.zeros((nrFolds, nrPat), bool)
      # testIndices = np.zeros((nrFolds, nrPat), bool)
      seed = 1
      skf = StratifiedKFold(n_splits = nrFolds, shuffle= True, random_state = seed)
      foldIndGen = skf.split(uniquePatCode,np.zeros(nrUniquePat), uniqueDiag)

      dpmObj = [0 for x in range(nrFoldsCurrProc)]
      # train the progression model on the training Data
      maxLikStagesLong = []
      stagingProbLong = []

      # acc, sens, spec, balAcc, #visitsClass1, #visitsClass2, #subjClass1, #subjClass2
      diagStatsCurrProc = np.zeros((nrFoldsCurrProc,8))
      diagClasses = np.sort(np.unique(filteredParams['diag']))

      for fld in range(nrFoldsCurrProc):
        foldIndex = foldInstances[fld]
        (trainIndicesUnq, testIndicesUnq) = [x for x in foldIndGen][foldIndex]
        trainIndices = np.in1d(filteredParams['partCode'], uniquePatCode[trainIndicesUnq])
        testIndices = np.in1d(filteredParams['partCode'], uniquePatCode[testIndicesUnq])

        print("foldIndex %d" % foldIndex)

        expNameCurrFold = 'dirDiag/%s/f%d' % (expName, foldIndex)
        dpmObj[fld] = dpmBuilder.generate(trainIndices, expNameCurrFold, filteredParams)

        dpmRes = dpmObj[fld].run(params['runPartDirDiag'])
        # dpmObj[fld].plotTrajSummary(dpmRes)
        ( _, _, stagingProbTrain, _,_, _) = dpmObj[fld].stageSubjects(trainIndices)

        (maxLikStages, maxStagesIndex, stagingProbTest, stagingLik,_, _) = \
          dpmObj[fld].stageSubjects(testIndices)

        # diagnosis between CTL and patients

        # first find optimal threshold using training data
        idealThresh = findIdealThresh(stagingProbTrain, filteredParams['diag'][trainIndices])
        diagStatsCurrProc[fld,0:6] = findDiagStatsGivenTh(
          stagingProbTest, filteredParams['diag'][testIndices], idealThresh)
        diagStatsCurrProc[fld, 6:8] = [np.sum(uniqueDiag[testIndicesUnq] == diagClasses[0]),
                                        np.sum(uniqueDiag[testIndicesUnq] == diagClasses[1])]

      savedData = dict(diagStatsCurrProc=diagStatsCurrProc, foldInstances=foldInstances)
      pickle.dump(savedData, open(procResFile[runIndex-1], 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    diagStats = None

  return diagStats

def evalStaging(dpmBuilder, expName, params):
  statsFile = 'matfiles/staging/%s/resAll.npz' % expName
  nrProcesses = params['nrProcesses']
  runIndex = params['runIndex']
  nrFolds = 10
  procResFile = ['matfiles/staging/%s/procRes_n%d_p%d.npz' % (expName, nrProcesses, p)
                 for p in range(1,nrProcesses+1)]

  if params['masterProcess']:
    if params['runPartStaging'][2] == 'R' or params['runPartStaging'][2] == 'L':
      # upEqStagesPercAll = np.zeros((nrFolds,1))
      # pFUgrBLAll = np.zeros((nrFolds,1))
      upEqStagesPercAll = []
      pFUgrBLAll = []
      timeDiffHardAll = []
      timeDiffSoftAll = []
      for p in range(nrProcesses):
        print(procResFile[p])
        if os.path.isfile(procResFile[p]):
          savedData = pickle.load(open(procResFile[p], 'rb'))

          for fld in range(len(savedData['foldInstances'])):
            upEqStagesPercAll += [np.mean(savedData['upEqStagesPercCurrProc'][fld])]
            pFUgrBLAll += [np.mean(savedData['pFUgrBLCurrProc'][fld])]

            timeDiffHardAll += [np.mean(savedData['timeDiffHardCurrProc'][fld])]
            timeDiffSoftAll += [np.mean(savedData['timeDiffSoftCurrProc'][fld])]

        else:
          raise IOError("file %s found" % procResFile[p])

      print(pFUgrBLAll)
      upEqStagesPerc = [np.mean(upEqStagesPercAll), np.std(upEqStagesPercAll)]
      pFUgrBL = [np.mean(pFUgrBLAll), np.std(pFUgrBLAll)]
      timeDiffHard = [np.mean(timeDiffHardAll), np.std(timeDiffHardAll)]
      timeDiffSoft = [np.mean(timeDiffSoftAll), np.std(timeDiffSoftAll)]

      # print('upEqStagesPercAll', upEqStagesPercCurrProc)
      # print('pFUgrBLAll', pFUgrBLAll)
      # print(ads)

      pass
    else:
      upEqStagesPerc = None
      pFUgrBL = None
      timeDiffHard = None
      timeDiffSoft = None
  else:
    if params['runPartStaging'][2] == 'R':
      filteredParams = filterDDSPA(params, params['excludeXvalidID'])
      # filter out patients that have only one visit,
      # otherwise the folds have very different numbers of patients due to later filtering of these
      moreThan1VisitIndices = np.array([np.sum(filteredParams['partCode'] == p) > 1 for p in filteredParams['partCode']])
      filteredParams = filterDDSPAIndices(filteredParams, moreThan1VisitIndices)

      foldInstances = allocateRunIndicesToProcess(nrFolds, nrProcesses, runIndex)
      nrFoldsCurrProc = len(foldInstances)

      (uniquePatCode, uniqueIndices) = np.unique(filteredParams['partCode'], return_index=True)
      nrUniquePat = uniquePatCode.shape[0]
      nrPat = filteredParams['diag'].shape[0]
      uniqueDiag = filteredParams['diag'][uniqueIndices]
      # trainIndices = np.zeros((nrFolds, nrPat), bool)
      # testIndices = np.zeros((nrFolds, nrPat), bool)
      seed = 1
      skf = StratifiedKFold(n_splits = nrFolds, shuffle= True, random_state = seed)
      foldIndGen = skf.split(uniquePatCode,np.zeros(nrUniquePat), uniqueDiag)

      dpmObj = [0 for x in range(nrFoldsCurrProc)]
      # train the progression model on the training Data
      maxLikStagesLong = []
      stagingProbLong = []

      upEqStagesPercStats = np.zeros((nrFoldsCurrProc,1))
      pFUgrBLStats = np.zeros((nrFoldsCurrProc,1))
      fldList = [x for x in foldIndGen]
      diagClasses = np.sort(np.unique(filteredParams['diag']))

      upEqStagesPercCurrProc = [0 for x in range(nrFoldsCurrProc)]
      pFUgrBLCurrProc = [0 for x in range(nrFoldsCurrProc)]
      timeDiffHardCurrProc = [0 for x in range(nrFoldsCurrProc)]
      timeDiffSoftCurrProc = [0 for x in range(nrFoldsCurrProc)]

      ## for testing only
      # dpmObj = [0 for x in range(nrFolds)]
      # for foldIndex in range(nrFolds):
      #   (trainIndicesUnq, testIndicesUnq) = fldList[foldIndex]
      #   trainIndices = np.in1d(filteredParams['partCode'], uniquePatCode[trainIndicesUnq])
      #   testIndices = np.in1d(filteredParams['partCode'], uniquePatCode[testIndicesUnq])
      #
      #   expNameCurrFold = 'staging/%s/f%d' % (expName, foldIndex)
      #   dpmObj[foldIndex] = dpmBuilder.generate(trainIndices, expNameCurrFold, filteredParams)
      #
      #   dpmRes = dpmObj[foldIndex].run(params['runPartStaging'])
      #   # dpmObj[fld].plotTrajectories(dpmRes)
      #
      #   (maxLikStagesTrain, _, stagingProbTrain, _, _) = dpmObj[foldIndex].stageSubjects(trainIndices)
      #
      #   (maxLikStages, maxStagesIndex, stagingProbTest, stagingLik, _) = dpmObj[foldIndex].stageSubjects(testIndices)
      #
      #   maxLikStagesLongTest = makeLongArray(maxLikStages, filteredParams['scanTimepts'][testIndices],
      #                                        filteredParams['partCode'][testIndices])
      #   print('fld %d #subj in testSet %d' % (foldIndex, len(maxLikStagesLongTest)))
      #
      # import pdb
      # pdb.set_trace()
      ### end testing

      for fld in range(nrFoldsCurrProc):
        foldIndex = foldInstances[fld]
        (trainIndicesUnq, testIndicesUnq) = fldList[foldIndex]
        trainIndices = np.in1d(filteredParams['partCode'], uniquePatCode[trainIndicesUnq])
        testIndices = np.in1d(filteredParams['partCode'], uniquePatCode[testIndicesUnq])
        exclStagingIndices = np.logical_not(np.in1d(filteredParams['diag'], params['excludeStaging']))
        testIndicesExclStag = np.logical_and(testIndices, exclStagingIndices)

        expNameCurrFold = 'staging/%s/f%d' % (expName, foldIndex)
        dpmObj[fld] = dpmBuilder.generate(trainIndices, expNameCurrFold, filteredParams)

        dpmRes = dpmObj[fld].run(params['runPartStaging'])
        #dpmObj[fld].plotTrajectories(dpmRes)

        (maxLikStagesTrain, _, stagingProbTrain, _,_, _) = dpmObj[fld].stageSubjects(trainIndices)

        (maxLikStagesTest, maxStagesIndex, stagingProbTest, stagingLik, tsStages, _) = dpmObj[fld].stageSubjects(testIndicesExclStag)

        maxLikStagesLongTest = makeLongArray(maxLikStagesTest, filteredParams['scanTimepts'][testIndicesExclStag],
                                         filteredParams['partCode'][testIndicesExclStag])
        stagingProbLongTest = makeLongArray(stagingProbTest, filteredParams['scanTimepts'][testIndicesExclStag],
                                         filteredParams['partCode'][testIndicesExclStag])
        ageAtScanLongTest = makeLongArray(filteredParams['ageAtScan'][testIndicesExclStag], filteredParams['scanTimepts'][testIndicesExclStag],
                                            filteredParams['partCode'][testIndicesExclStag])

        # staging consistency
        upEqStagesPercCurrProc[fld], pFUgrBLCurrProc[fld], timeDiffHardCurrProc[fld], \
        timeDiffSoftCurrProc[fld] = \
          stagingConsistency(maxLikStagesLongTest, stagingProbLongTest, tsStages, ageAtScanLongTest)

         # import pdb
        # pdb.set_trace()  # find out why the staging consistency is low in model 3 but high in model 1
        # visualizeStagingProb(stagingProbLongTest, params['tsStages'], params['plotTrajParams'], expName)

        savedData = dict(upEqStagesPercCurrProc=upEqStagesPercCurrProc,
               pFUgrBLCurrProc=pFUgrBLCurrProc, timeDiffHardCurrProc=timeDiffHardCurrProc,
               timeDiffSoftCurrProc=timeDiffSoftCurrProc, foldInstances=foldInstances)
      pickle.dump(savedData, open(procResFile[runIndex - 1], 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    upEqStagesPerc = None
    pFUgrBL = None
    timeDiffHard = None
    timeDiffSoft = None

  return upEqStagesPerc, pFUgrBL, timeDiffHard, timeDiffSoft

def stagingConsistency(maxLikStages, stagingProb, tsStages, ageAtScan):
  decrPairsCount = 0
  totalPairsCount = 0
  upEqStagesAll = []
  timeDiffHard = []
  # print(maxLikStages)

  nrStages = stagingProb[0].shape[1]
  pFUgrBLgBL = np.zeros(nrStages) # p(FU > BL | BL)
  pFUgrBLall = []
  timeDiffSoft = []
  # soft metrics

  for p in range(len(maxLikStages)):
    for tmpt in range(len(maxLikStages[p]) - 1):

      ######## hard metrics ##############

      # staging consistency
      if maxLikStages[p][tmpt + 1] >= maxLikStages[p][tmpt]:
        upEqStagesAll += [1]
      else:
        upEqStagesAll += [0]

      # time difference metric
      predDiffHard = maxLikStages[p][tmpt+1] - maxLikStages[p][tmpt]
      actualDiffHard = ageAtScan[p][tmpt+1] - ageAtScan[p][tmpt]
      timeDiffHard += [np.abs(predDiffHard - actualDiffHard)]

      ######## soft metrics ##############

      pXi = stagingProb[p][tmpt,:]
      pXii = stagingProb[p][tmpt+1,:]
      # staging consistency
      if any([math.isnan(x) for x in pXi]) or  any([math.isnan(x) for x in pXii]):
        print('pXi p%d t%d' % (p, tmpt),pXi)
        print('pXii',pXii)
        raise Exception("pXi and pXii contain nans")

      for st in range(nrStages):
        pFUgrBLgBL[st] = np.sum(pXii[st:])

      assert(not any([math.isnan(x) for x in pFUgrBLgBL]))
      pFUgrBLall += [sum(np.multiply(pXi, pFUgrBLgBL))] # posterior p(Followup stage >= BL stage)

      # time difference metric
      predDiffSoft = np.sum(np.multiply(pXii, tsStages)) - np.sum(np.multiply(pXi, tsStages))
      actualDiffSoft = ageAtScan[p][tmpt + 1] - ageAtScan[p][tmpt]
      timeDiffSoft += [np.abs(predDiffSoft - actualDiffSoft)]

      pass

  #pFUgrBL = [mean(pFUgrBLall), std(pFUgrBLall)];
  if any([math.isnan(x) for x in pFUgrBLall]):
    print(pFUgrBLall)
    raise Exception("pFUgrBLall contain nans")

  #pFUgrBL = np.mean(pFUgrBLall)

  # print(maxLikStages)
  # print(upEqStagesPercAll, pFUgrBL)
  # print(adsa)

  return upEqStagesAll, pFUgrBLall, timeDiffHard, timeDiffSoft

def allocateRunIndicesToProcess(nrExperiments, nrProcesses, runIndex):
  indicesCurrProcess = [x for x in range(nrExperiments)][(runIndex-1):nrExperiments:nrProcesses]
  return indicesCurrProcess
