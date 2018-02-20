import scipy.io as sio
import sys
import pickle

from DisProgBuilder import *
from evaluationFramework import *
from DEM import *
from aux import *

minTstaging = -20
maxTstaging = 20
nrStages = 200
params = {'tsStages' : np.linspace(minTstaging, maxTstaging, num=nrStages)}
plotTrajParams = {'diagLabels' : ['CTL', 'MCI', 'AD'],
                  'diagColors' : ['g', 'y', 'r'],
                  'xLim' : (-20, 30) }


def main(runIndex, nrProcesses, modelToRun):
  blData = sio.loadmat('../data/ADNI/ADNIdata_Baseline.mat')

  # for 12m and 24m follow-up data, for longitudinal consistency
  fuData = sio.loadmat('../data/ADNI/ADNILongitTimeptsPartCodes.mat')

  # for classification between CN-stable vs CN-converters, actually contains baseline data, split into groups
  converterData = sio.loadmat('../data/ADNI/ADNI_long_consistency_data.mat')

  print(blData.keys(), fuData.keys(), converterData.keys())

  np.random.seed(7)

  expName = 'adni'

  # ATTENTION: The following indices use MATLAB 0-indexing, subtract 1 when doing in python

  #selectedBiomk = np.array([0,1,2,3,4,5,6,7,8,9,10,11])
  selectedBiomk = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11])

  labels = ['T-TAU', 'ABETA142', 'P-TAU', 'Ventricles',
  'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform',
  'Mid Temporal', 'ADAS13', 'MMSE', 'RAVLT']#, 'Brain atrophy', 'Hippo. atrophy']


  params['data'] = fuData['data0m_12m_24m'][:,selectedBiomk]
  params['diag'] = np.array([x[0] for x in fuData['diag0m_12m_24m']])
  params['blData'] = blData['EBMdataBL'][:,selectedBiomk]
  params['blDiag']  = np.array([x[0] for x in blData['EBMdxBL']])
  params['blDataPartCode'] = np.array([x[0][0] for x in blData['ADNIdataBL'][1:,1]])
  params['labels'] = labels
  params['scanTimepts'] = np.array([x[0] for x in fuData['scanTimepoint']])
  params['partCode'] = np.array([x[0] for x in fuData['participantCode']])
  params['ageAtScan'] = np.array([x[0] for x in fuData['ageAtScan']])
  params['cnConvRIDs'] = converterData['cnConvRIDs'][0][0]
  params['cnStableRIDs'] = converterData['cnStableRIDs'][0][0]
  params['mciConvRIDs'] = converterData['mciConvRIDs'][0][0]
  params['mciStableRIDs'] = converterData['mciStableRIDs'][0][0]
  params['lengthScaleFactors'] = np.ones(len(selectedBiomk))

  params['runIndex'] = runIndex
  params['nrProcesses'] = nrProcesses
  params['modelToRun'] = modelToRun

  nrBiomk = len(labels)

  print(nrBiomk)
  biomkProgDir = np.zeros(nrBiomk)
  #biomkProgDir[[0,2,3,9,12,13]] = INCR
  biomkProgDir[[0, 2, 3, 9,]] = INCR
  biomkProgDir[[1,4,5,6,7,8,10,11]] = DECR
  biomkProgDir = biomkProgDir[selectedBiomk]

  params['data'] = makeAllSameProgDir(params['data'], biomkProgDir, uniformDir)
  params['blData'] = makeAllSameProgDir(params['blData'], biomkProgDir, uniformDir)

  nrRows = int(np.sqrt(nrBiomk) * 0.95)
  nrCols = int(np.ceil(float(nrBiomk) / nrRows))
  assert(nrRows * nrCols >= nrBiomk)

  plotTrajParams['modelCol'] = 'r' # red
  plotTrajParams['xLim'] = [-20, 30]
  plotTrajParams['axisPos'] = [0.06, 0.1, 0.9, 0.75]
  plotTrajParams['legendPos'] = (0.5, 1.1)
  plotTrajParams['legendPosSubplotsPcaAd'] = (0.5, 0)
  plotTrajParams['legendCols'] = 4
  plotTrajParams['nrRows'] = nrRows
  plotTrajParams['nrCols'] = nrCols
  plotTrajParams['trajAlignMaxWinSize'] = (900, 700)
  plotTrajParams['trajPcaAdMaxWinSize'] = (1200, 500)
  plotTrajParams['axisHeightChopRatio'] = 0.8
  plotTrajParams['expName'] = expName
  plotTrajParams['diagColors'] = ['b', 'y', 'r']
  params['plotTrajParams'] = plotTrajParams

  params['runPartMain'] = ['L', 'L', 'I', 'I']
  params['runPartStaging'] = ['L', 'L', 'R']
  params['runPartConvPred'] = ['L', 'L', 'L']

  params['masterProcess'] = runIndex == 0

  if params['masterProcess']:
    params['runPartMain'] = ['L', 'L', 'I', 'I']
    params['runPartStaging'] = ['R', 'R', 'L']
    params['runPartConvPred'] = ['R', 'R', 'L']

  runAllExpFunc = runAllExpADNI
  modelNames, res = runModels(params, expName, modelToRun, runAllExpFunc)

  if params['masterProcess']:
    printResADNI(modelNames, res)

def printResADNI(modelNames, res):
  nrModels = len(modelNames)
  upEqStagesPerc = np.zeros((nrModels,2))
  pFUgrBLAll = np.zeros((nrModels,2))
  timeDiffHard = np.zeros((nrModels,2))
  timeDiffSoft = np.zeros((nrModels, 2))
  periods = ['12m','24m','36m']
  convPred = {'12m' : np.zeros((nrModels,2)), '24m' : np.zeros((nrModels,2)),
              '36m' : np.zeros((nrModels,2))}
  nrPeriods = len(periods)
  for m in range(nrModels):
    upEqStagesPerc[m,:] = res[m]['upEqStagesPerc']
    pFUgrBLAll[m,:] = res[m]['pFUgrBLAll']

    timeDiffHard[m, :] = res[m]['timeDiffHard']
    timeDiffSoft[m, :] = res[m]['timeDiffSoft']

    for p in range(nrPeriods):
      convPred[periods[p]][m,:] = [res[m]['convPredStats']['mean'][p,0],
                          res[m]['convPredStats']['std'][p, 0]]

  np.set_printoptions(precision=4)
  print(modelNames)
  print('upEqStagesPerc',arrayToStrNoBrackets(upEqStagesPerc))
  print('pFUgrBLAll',arrayToStrNoBrackets(pFUgrBLAll))
  print('timeDiffHard', arrayToStrNoBrackets(timeDiffHard))
  print('timeDiffSoft', arrayToStrNoBrackets(timeDiffSoft))

  print('convPred[12m]', arrayToStrNoBrackets(convPred['12m']))
  print('convPred[24m]', arrayToStrNoBrackets(convPred['24m']))
  print('convPred[36m]', arrayToStrNoBrackets(convPred['36m']))

  formalLabels = ['DEM - Standard Alignment', 'DEM - Optimised Alignment']

  print('adni staging')
  for m in range(nrModels):
    print('  %s & %.2f $\pm$ %.2f & %.2f $\pm$ %.2f & %.2f $\pm$ %.2f & %.2f $\pm$ %.2f\\\\' %
          (formalLabels[m], upEqStagesPerc[m,0], upEqStagesPerc[m,1],
          pFUgrBLAll[m, 0], pFUgrBLAll[m, 1],
          timeDiffHard[m, 0], timeDiffHard[m, 1],
          timeDiffSoft[m, 0], timeDiffSoft[m, 1]))

  print('adni conv pred')
  for m in range(nrModels):
    print('  %s & %.2f $\pm$ %.2f & %.2f $\pm$ %.2f & %.2f $\pm$ %.2f\\\\' %
          (formalLabels[m], convPred['12m'][m,0], convPred['12m'][m,1],
           convPred['24m'][m, 0], convPred['24m'][m,1],
           convPred['36m'][m, 0], convPred['36m'][m, 1]))
  pass

def runAllExpADNI(params, expName, dpmBuilder):
  """ runs all experiments"""

  res = {}

  params['patientID'] = AD
  params['excludeID'] = -1
  params['excludeXvalidID'] = -1
  params['excludeStaging'] = [-1]
  params['anchorID'] = MCI

  # run if this is the master   process or nrProcesses is 1
  unluckyProc = (np.mod(params['currModel'] - 1, params['nrProcesses']) == params['runIndex'] - 1)
  unluckyOrNoParallel = unluckyProc or (params['nrProcesses'] == 1) or params['masterProcess']

  if unluckyOrNoParallel:
    dpmObj, res['std'] = runStdDPM(params, expName, dpmBuilder, params['runPartMain'])

  res['upEqStagesPerc'], res['pFUgrBLAll'], res['timeDiffHard'], res['timeDiffSoft'] = \
    evalStaging(dpmBuilder, expName, params)
  res['convPredStats'] = evalConversionADNI(dpmBuilder, expName, params)

  res['cogCorr'] = crossValidAndCorrCog(dpmBuilder, expName, params)
  #res['predBiomk'] = predBiomk(dpmBuilder, expName, params) # mainly for voxelwise model, but extensible for DEM and EBM
  cvNonOverlapFolds(dpmBuilder, expName, params)

  # print(res)

  return res

def evalConversionADNI(dpmBuilder, expName, params):
  """
  Perform cross-validated conversion prediction using the given DPM.
  Cross-validation is done on EBMdataBL (285 entries), and at each fold the training data
  from EBMdataBL is intersected with the longitudinal data (data0m_12m_24m), which is much smaller.

  Parameters
  ----------
  dpmBuilder
  expName
  params

  Returns
  -------

  """
  statsFile = 'matfiles/adniConv/%s/stats.npz' % expName
  nrProcesses = params['nrProcesses']
  runIndex = params['runIndex']
  nrFolds = 10
  procResFile = ['matfiles/adniConv/%s/procRes_n%d_p%d.npz' % (expName, nrProcesses, p)
                 for p in range(1,nrProcesses+1)]

  periods = [12,24,36]
  grStrs = ['cn', 'mci']
  nrGroups = len(grStrs)
  nrPeriods = len(periods)
  grNr = 1
  STABLE_DIAG = 1
  CONV_DIAG = 2

  if params['masterProcess']:
    if params['runPartConvPred'][2] == 'R' or params['runPartConvPred'][2] == 'L':
      diagStatsAll = np.zeros((nrFolds, nrPeriods, 6))
      for p in range(nrProcesses):
        if os.path.isfile(procResFile[p]):
          savedData = pickle.load(open(procResFile[p], 'rb'))
          diagStatsAll[savedData['foldInstances'], :,:] = savedData['diagStatsCurrProc']
        else:
          raise IOError("file %s found" % procResFile[p])

      diagStats = {'mean': np.mean(diagStatsAll, axis=0), 'std': np.std(diagStatsAll, axis=0)}
      pass
    else:
      diagStats = None
  else:
    if params['runPartConvPred'][2] == 'R':
      foldInstances = allocateRunIndicesToProcess(nrFolds, nrProcesses, runIndex)
      nrFoldsCurrProc = len(foldInstances)

      blPartCode = params['blDataPartCode']
      nrPart = params['partCode'].shape[0]
      diag = params['blDiag']
      partCodeAD = blPartCode[diag == AD]
      blDiagCtlMciIndices = np.logical_or(diag == CTL, diag == MCI)
      blPartCodeCtlMci = blPartCode[blDiagCtlMciIndices]

      # fuTrainIndices = np.zeros((nrFolds, nrPart), bool)
      # fuTestIndices = np.zeros((nrFolds, nrPart), bool)
      seed = 2
      skf = StratifiedKFold(n_splits = nrFolds, shuffle = True, random_state = seed)
      foldIndGen = skf.split(blPartCodeCtlMci,np.zeros(np.sum(blDiagCtlMciIndices)),
                             diag[blDiagCtlMciIndices])

      dpmObj = [0 for x in range(nrFoldsCurrProc)]
      # train the progression model on the training Data
      maxLikStagesLong = []
      stagingProbLong = []

      diagStatsCurrProc = np.zeros((nrFoldsCurrProc, nrPeriods, 6)) # acc, sens, spec, balAcc, #visitsClass1, #visitsClass2, #subjClass1,
        #subjClass2
      confMatCurrProc = np.zeros((nrFoldsCurrProc,2,2))

      for fld in range(nrFoldsCurrProc):
        foldIndex = foldInstances[fld]
        (blTrainIndicesCtlMci, blTestIndicesCtlMci) = [x for x in foldIndGen][foldIndex]
        blTrainPartCodeCtlMci = blPartCodeCtlMci[blTrainIndicesCtlMci]
        blTestPartCodeCtlMci = blPartCodeCtlMci[blTestIndicesCtlMci]
        blTrainPartCode = np.concatenate((blTrainPartCodeCtlMci, partCodeAD))

        # print(np.concatenate((uniquePartCodeCtlMci[trainIndicesUnq], partCodeAD)),'----')
        # print(uniquePartCodeCtlMci[trainIndicesUnq],'-----', partCodeAD)
        fuTrainIndices = np.in1d(params['partCode'], blTrainPartCode)
        fuTestIndices = np.in1d(params['partCode'], blTestPartCodeCtlMci)

        blTrainIndices = np.in1d(params['blDataPartCode'], blTrainPartCode)
        blTestIndices = np.in1d(params['blDataPartCode'], blTestPartCodeCtlMci)

        # build DPM on training data
        expNameCurrFold = 'adniConv/%s/f%d' % (expName, foldIndex)
        params['excludeID'] = -1
        dpmObj[fld] = dpmBuilder.generate(fuTrainIndices, expNameCurrFold, params)
        dpmRes = dpmObj[fld].run(params['runPartConvPred'])
        #dpmObj[foldIndex].plotTrajSummary(dpmRes)
        dpmSamplesRes = dpmObj[fld].genPosteriorSamples(dpmRes)

        for periodNr in range(nrPeriods):
          stablePartCodeCurrPeriod = params['%sStableRIDs' % grStrs[grNr]]['m%d' % periods[periodNr]][0]
          convPartCodeCurrPeriod = params['%sConvRIDs' % grStrs[grNr]]['m%d' % periods[periodNr]][0]
          partCodeCurrPeriod = np.concatenate((stablePartCodeCurrPeriod, convPartCodeCurrPeriod))

          # create stable datasets
          stableIndCurrPeriod = np.in1d(params['blDataPartCode'], stablePartCodeCurrPeriod)
          trainStableIndCurrPeriod = np.logical_and(stableIndCurrPeriod, blTrainIndices)
          testStableIndCurrPeriod = np.logical_and(stableIndCurrPeriod, blTestIndices)
          trainStableDataCurrPeriod = params['blData'][trainStableIndCurrPeriod,:]
          testStableDataCurrPeriod = params['blData'][testStableIndCurrPeriod, :]

          # create converter datasets
          convIndCurrPeriod = np.in1d(params['blDataPartCode'], convPartCodeCurrPeriod)
          trainConvIndCurrPeriod = np.logical_and(convIndCurrPeriod, blTrainIndices)
          testConvIndCurrPeriod = np.logical_and(convIndCurrPeriod, blTestIndices)
          trainConvDataCurrPeriod = params['blData'][trainConvIndCurrPeriod, :]
          testConvDataCurrPeriod = params['blData'][testConvIndCurrPeriod, :]

          # merge stable and converter datasets together and assign diagnosis (or rather prognosis)
          trainDiagCurrPeriod = np.concatenate((STABLE_DIAG * np.ones(trainStableDataCurrPeriod.shape[0]),
                                           CONV_DIAG * np.ones(trainConvDataCurrPeriod.shape[0])))
          testDiagCurrPeriod = np.concatenate((STABLE_DIAG * np.ones(testStableDataCurrPeriod.shape[0]),
                                           CONV_DIAG * np.ones(testConvDataCurrPeriod.shape[0])))
          trainDataCurrPeriod = np.concatenate((trainStableDataCurrPeriod, trainConvDataCurrPeriod))
          testDataCurrPeriod = np.concatenate((testStableDataCurrPeriod, testConvDataCurrPeriod))
          #trainDataCurrPeriodNorm = dpmObj[foldIndex].getDataZ(trainDataCurrPeriod)
          #testDataCurrPeriodNorm = dpmObj[foldIndex].getDataZ(testDataCurrPeriod)

          #print(testStableDataCurrPeriod.shape, testConvDataCurrPeriod.shape)
          minSizeDatasetsSatisfied = (trainStableDataCurrPeriod.shape[0] > 1 and trainConvDataCurrPeriod.shape[0] > 1 and
              testStableDataCurrPeriod.shape[0] > 1 and testConvDataCurrPeriod.shape[0] > 1)
          assert minSizeDatasetsSatisfied
          if minSizeDatasetsSatisfied:
            (maxLikStagesTrain, _, stagingProbTrain, _,tsStagesTrain) = dpmObj[fld].stageSubjectsData(trainDataCurrPeriod)
            (maxLikStagesTest, _, stagingProbTest, _, tsStagesTest) = dpmObj[fld].stageSubjectsData(testDataCurrPeriod)

            #print((params['blData'][:,0].min(),params['blData'][:,0].max()),
            #      (params['data'][:,0].min(),params['data'][:,0].max()))
            #print(asds)
            #(_, _, stagingProbMock, _, _) = dpmObj[foldIndex].stageSubjects(np.arange(0, len(params['diag']), 1))

            idealThreshIndex = findIdealThresh(stagingProbTrain, trainDiagCurrPeriod)
            diagStatsCurrProc[fld, periodNr, 0:6] = findDiagStatsGivenTh(stagingProbTest, testDiagCurrPeriod,
                                                                idealThreshIndex)

            #plotTrajSubfigWithData(dpmObj[foldIndex].ts, dpmObj[foldIndex].xsZ, dpmSamplesRes['tsSamples'],
            #  dpmSamplesRes['xsSamples'], dpmSamplesRes['badSamples'], params['labels'], params['plotTrajParams'],
            #   dpmObj[foldIndex].getDataZ(testDataCurrPeriod), testDiagCurrPeriod, maxLikStagesTest, thresh=tsStagesTrain[idealThreshIndex])

            # if periodNr == 1 and foldIndex > 2:
            #   plotTrajSubfigWithData(dpmObj[foldIndex].ts, dpmObj[foldIndex].xsZ, dpmSamplesRes['tsSamples'],
            #     dpmSamplesRes['xsSamples'], dpmSamplesRes['badSamples'], params['labels'], params['plotTrajParams'],
            #      dpmObj[foldIndex].getDataZ(trainDataCurrPeriod), trainDiagCurrPeriod, maxLikStagesTrain, thresh=tsStagesTrain[idealThreshIndex])

            #print(asds)
            pass

      savedData = dict(diagStatsCurrProc=diagStatsCurrProc, foldInstances=foldInstances)
      pickle.dump(savedData, open(procResFile[runIndex-1], 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    diagStats = None

  return diagStats


if __name__ == '__main__':
  runIndex = int(sys.argv[1])
  nrProcesses = int(sys.argv[2])
  modelToRun = int(sys.argv[3])
  main(runIndex, nrProcesses, modelToRun)