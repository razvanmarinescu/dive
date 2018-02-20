import os
# from socket import gethostname
import os.path
# import nibabel as nib
# import copy
from voxelDPM import *
from VDPMLinear import *
from aux import *



def calcProbControlFromExpo(stage, muExpoCTL, muExpoPAT, stageLowerLim, stageUpperLim):

  probControl = scipy.stats.expon.pdf(stage-stageLowerLim, scale=muExpoCTL-stageLowerLim) / \
        (scipy.stats.expon.pdf(stage-stageLowerLim, scale=muExpoCTL-stageLowerLim) +
         scipy.stats.expon.pdf(stageUpperLim - stage, scale=muExpoPAT-stageLowerLim))

  return probControl

def generateDiag(dpsCross):
  nrSubjCross = dpsCross.shape[0]
  diagPrecDef = 0.4
  controlDiagPrec = diagPrecDef
  patientDiagPrec = diagPrecDef
  minDps = np.min(dpsCross)
  maxDps = np.max(dpsCross)
  #dpsUpperLim = upperAgeLim # after this dps limit limit almost all of diags will be patient
  # precision values they cannot be 1(perfect recision) as the exponential distribution is not well - defined anymore
  assert (controlDiagPrec != 1 and patientDiagPrec != 1)
  muScale = 1
  # multiplying the mean with nrTimepts scales perfectly to more biomk, tested on 18 / 03 / 2016
  muExpoCTL = minDps + muScale * (maxDps - minDps) * (1 - controlDiagPrec**(1 / 2))
  muExpoPAT = minDps + muScale * (maxDps - minDps) * (1 - patientDiagPrec**(1 / 2))
  diagCross = CTL * np.ones(nrSubjCross, int)
  probControl = np.zeros(nrSubjCross, float)
  for s in range(nrSubjCross):
    # generate diag
    dpsCurr = dpsCross[s]
    probControl[s] = calcProbControlFromExpo(dpsCurr, muExpoCTL, muExpoPAT, minDps, maxDps)

    if np.random.rand(1, 1) > probControl[s]:
      diagCross[s] = AD

  # plot probControl over dps's
  nrStages = 100
  stageRange = np.linspace(minDps, maxDps, nrStages)
  probControlStages = np.zeros(nrStages, float)
  for st in range(nrStages):
    probControlStages[st] = calcProbControlFromExpo(stageRange[st], muExpoCTL, muExpoPAT, minDps, maxDps)

  assert not np.isnan(probControl).any()

  # print('dpsCross', dpsCross)
  # print('probControl', probControl)
  # print(stageRange, probControlStages)
  # print(muExpoCTL, muExpoPAT)
  # print(minDps, maxDps)
  # pl.plot(stageRange, probControlStages)
  # pl.show()

  return diagCross

def generateClustData(nrSubjLong, nrBiomk, nrClust, nrTimepts, trajFunc, thetasTrue,
  thetasPerturbed, clustAssignTrueB, lowerAgeLim, upperAgeLim, covSubjShifts,avgStdScaleFactor, fileName,
  forceRegenerate, makeThetaIdentifFunc, localParams):

  if os.path.isfile(fileName) and not forceRegenerate:
    dataStruct = pickle.load(open(fileName, 'rb'))
    dataCross = dataStruct['dataCross']
    diagCross = dataStruct['diagCross']
    scanTimeptsCross = dataStruct['scanTimeptsCross']
    partCodeCross = dataStruct['partCodeCross']
    ageAtScanCross = dataStruct['ageAtScanCross']
    trueParams = dataStruct['trueParams']

  else:

    np.random.seed(1)
    # generate subject data
    subShiftsLongTrue = np.array([np.random.multivariate_normal(
      [1, 0], covSubjShifts) for s in range(nrSubjLong)])
    subShiftsLongTrue[:,0] = np.abs(subShiftsLongTrue[:,0]) # ensure alphas > 0
    nrSubjCross = nrTimepts * nrSubjLong

    ageAtBlScanLong = np.array([np.random.uniform(lowerAgeLim,upperAgeLim) for s in range(nrSubjLong)])
    ageAtScanCross = np.zeros(nrSubjCross, float)

    dataCross = np.zeros((nrSubjCross, nrBiomk), float)
    subShiftsCrossTrue = np.zeros((nrSubjCross,2), float)

    partCodeCross = np.zeros(nrSubjCross, float)
    partCodeLong = np.array(range(nrSubjLong)) # unique id for every participant
    scanTimeptsCross = np.zeros(nrSubjCross, float)

    counter = 0
    long2crossInd = np.zeros(nrSubjCross, int)

    for s in range(nrSubjLong):
      for tp in range(nrTimepts):
        # get currTimept, age at curr Timepints, and partCodeCross
        partCodeCross[counter] = partCodeLong[s]
        scanTimeptsCross[counter] = tp
        ageAtScanCross[counter] = ageAtBlScanLong[s] + tp # add one year at each timepoint
        subShiftsCrossTrue[counter,:] = subShiftsLongTrue[s,:]
        long2crossInd[counter] = s

        counter += 1
    # generate data - find dps from age
    dpsCross = ageAtScanCross * subShiftsCrossTrue[:, 0] + \
                        subShiftsCrossTrue[:, 1]  # disease progression score

    diagCross = generateDiag(dpsCross)
    print('diagCross', diagCross)
    assert np.unique(diagCross).shape[0] >= 2
    # print(adsa)

    diagLongFirstScan = diagCross[scanTimeptsCross == 0]

    meanAgeCTL = np.mean(ageAtScanCross[diagCross == CTL], 0)
    stdAgeCTL = np.std(ageAtScanCross[diagCross == CTL], 0)
    ageAtScanCross = (ageAtScanCross - meanAgeCTL) / stdAgeCTL

    longAgeAtScan = ageAtScanCross[scanTimeptsCross == 0]
    longAge1array = [np.concatenate((x.reshape(-1, 1), np.ones(x.reshape(-1, 1).shape)), axis=1) for x in longAgeAtScan]
    ageFirstVisitLong1array = np.array([s[0, :] for s in longAge1array])
    assert(ageFirstVisitLong1array.shape[1] == 2)

    # print('subShiftsLongTrue[:20]', subShiftsLongTrue[:20])
    # print('muAge sigma_Age', meanAgeCTL, stdAgeCTL)
    # print(asdas)

    # make the subject shifts and thetas identifiable, use same trans as in VDPM
    subShiftsLongTrue, shiftTransform = VoxelDPM.makeShiftsIdentif(
      subShiftsLongTrue, ageFirstVisitLong1array, diagLongFirstScan)

    # thetasTrue = makeThetaIdentifFunc(thetasTrue, shiftTransform)
    # thetasPerturbed = makeThetaIdentifFunc(thetasPerturbed, shiftTransform)

    # print('shiftTransform', shiftTransform)
    # print('subShiftsLongTrue[:20]', subShiftsLongTrue[:20])
    # print('muAge sigma_Age', meanAgeCTL, stdAgeCTL)
    # print(asdas)

    # # make shifts correspond as we Z-score the age, for setting a prior on them
    # subShiftsLongTrue[:, 0] /= stdAgeCTL # alpha = alpha / sigma_N
    # subShiftsLongTrue[:, 1] -= subShiftsLongTrue[:, 0]*meanAgeCTL # beta = beta - mu_N*alpha/sigma_N

    # print('subShiftsLongTrue[:20]', subShiftsLongTrue[:20])

    dpsLong = VoxelDPM.calcDps(subShiftsLongTrue, ageFirstVisitLong1array)
    dpsCTL = dpsLong[diagLongFirstScan == CTL]
    muCTL = np.mean(dpsCTL)
    sigmaCTL = np.std(dpsCTL)
    # print('muCTL', muCTL, 'sigmaCTL', sigmaCTL)

    subShiftsCrossTrue = subShiftsLongTrue[long2crossInd]
    dpsCross = ageAtScanCross * subShiftsCrossTrue[:, 0] + \
                        subShiftsCrossTrue[:, 1]  # disease progression score

    # print('dpsCross[diagCross == CTL][:20]', dpsCross[diagCross == CTL][:20])
    # print(asdas)
    print('subShiftsLongTrue', subShiftsLongTrue)
    print('ageAtScanCross', ageAtScanCross)
    print('np.abs(muCTL)', np.abs(muCTL))
    print('np.abs(sigmaCTL)', np.abs(sigmaCTL))
    assert(np.abs(muCTL < 0.1))
    assert (np.abs(sigmaCTL - 1) < 0.1)

    # midPt2 = trajFunc((upperAgeLim + lowerAgeLim)/2, thetasTrue[2,:])
    # midPt1 = trajFunc((upperAgeLim + lowerAgeLim)/2, thetasTrue[1,:])

    # set the variance proportional to the difference between the true lines
    #avgVar = np.abs(midPt1 - midPt2)**2
    # variancesFromPerturbedTrue = np.array([avgVar for i in range(nrBiomk)])
    #stdsFromPerturbedTrue = np.sqrt(variancesFromPerturbedTrue)
    avgStdFromPerturbedTrue = 0.5*avgStdScaleFactor

    for b in range(nrBiomk):
      fsCurrS = trajFunc(dpsCross, thetasPerturbed[b, :])
      dataCross[:, b] = np.random.randn(nrSubjCross) * avgStdFromPerturbedTrue + fsCurrS

    clustProbBCtrue = np.zeros((nrBiomk, nrClust), float)
    for b in range(nrBiomk):
      clustProbBCtrue[b,clustAssignTrueB[b]] = 1

    clustProbColNormBCtrue = clustProbBCtrue / np.sum(clustProbBCtrue,axis=0)[None, :]

    # estimate the true variances as the variancesPerturbedTrue + perturbation effect.
    # in practice, just estimate their variance from the sample of data points
    variancesTrue = np.zeros(nrClust,float)
    for c in range(nrClust):
      fsFromTrueTheta = trajFunc(dpsCross, thetasTrue[c, :])
      sqErrors = np.power(dataCross[:,clustAssignTrueB == c] - fsFromTrueTheta[:,None], 2)
      variancesTrue[c] = np.sum(sqErrors)/(sqErrors.shape[0]*sqErrors.shape[1])

    thetasPerturbedClust = [thetasPerturbed[clustAssignTrueB == c] for c in range(nrClust)]
    trueParams = dict(thetas=thetasTrue, subShiftsLong=subShiftsLongTrue,
      subShiftsCross=subShiftsCrossTrue, variances=variancesTrue, clustAssignB=clustAssignTrueB,
      thetasPerturbed=thetasPerturbed, thetasPerturbedClust=thetasPerturbedClust)

    print('nrClust', nrClust)
    print('clustProbColNormBCtrue', clustProbColNormBCtrue.shape)

    plotterObj = PlotterVDPM.PlotterVDPMSynth()
    plotterObj.plotTrajSubfigWithDataRandPoints(dataCross, diagCross, dpsCross, thetasTrue,
                                     variancesTrue, clustProbColNormBCtrue, localParams['plotTrajParams'], trajFunc,
                                     replaceFigMode=False,
                                     thetasSamplesClust =thetasPerturbedClust)

    dataStruct = dict(dataCross=dataCross, diagCross=diagCross, scanTimeptsCross=scanTimeptsCross,
      partCodeCross=partCodeCross, ageAtScanCross=ageAtScanCross,
      trueParams=trueParams)

    pickle.dump(dataStruct, open(fileName, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

  # import pdb
  # pdb.set_trace()

  meanAgeCTL = np.mean(ageAtScanCross[diagCross == CTL], 0)
  stdAgeCTL = np.std(ageAtScanCross[diagCross == CTL], 0)
  ageAtScanCrossZ = (ageAtScanCross - meanAgeCTL) / stdAgeCTL

  assert (not np.any(np.isnan(dataCross)))

  localParams['data'] = dataCross
  localParams['diag'] = diagCross
  localParams['scanTimepts'] = scanTimeptsCross
  localParams['partCode'] = partCodeCross
  localParams['ageAtScan'] = ageAtScanCrossZ
  localParams['trueParams'] = trueParams
  localParams['trueNrClust'] = nrClust # set how many clusters to fit
  localParams['trajFunc'] = trajFunc


  return localParams


def shuffleThetas(thetasTrueCurr):
  np.random.seed(1)
  thetasTrueCurr[:, 2] = np.random.permutation(thetasTrueCurr[:, 2])
  thetasTrueCurr[:, 1] = np.random.permutation(thetasTrueCurr[:, 1])
  return thetasTrueCurr

def generateThetas(nrClustToGenCurr, trajMinLowerLim, trajMinInterval,
  slopeLowerLim, slopeInterval, dpsLowerLimit, dpsInterval):

  # generate the theta parameters for the curr set of clusters, vary slopes, centers
  # and lower limits, keep the upper limit always zero (i.e. modelling that the data is z-scored)
  thetasTrueCurr = np.zeros((nrClustToGenCurr, 4), float)
  for c2 in range(nrClustToGenCurr):
    trajMin = trajMinLowerLim + trajMinInterval * c2 / nrClustToGenCurr
    assert trajMin < 0
    slopeCurr = slopeLowerLim + slopeInterval * c2 / nrClustToGenCurr
    thetasTrueCurr[c2, :] = [-trajMin, -slopeCurr * 4 / trajMin,
      dpsLowerLimit + dpsInterval * c2 / nrClustToGenCurr, trajMin]

  print('thetasTrueCurr', thetasTrueCurr)
  # print(adas)
  # shuffle their centers and slopes, otherwise the early clusters will always have low slopes
  thetasTrueCurr = shuffleThetas(thetasTrueCurr)

  return thetasTrueCurr

def runAllExpSynth(params, expName, dpmBuilder, compareTrueParamsFunc = None):
  """ runs all experiments"""

  res = {}

  params['patientID'] = AD
  params['excludeID'] = -1
  params['excludeXvalidID'] = -1
  params['excludeStaging'] = [-1]

  # run if this is the master   process or nrProcesses is 1
  unluckyProc = (np.mod(params['currModel'] - 1, params['nrProcesses']) == params['runIndex'] - 1)
  unluckyOrNoParallel = unluckyProc or (params['nrProcesses'] == 1) or params['masterProcess']

  dpmBuilder.setPlotter(PlotterVDPM.PlotterVDPMSynth())
  dpmObjStd, res['std'] = evaluationFramework.runStdDPM(params, expName, dpmBuilder,
    params['runPartMain'])

  if 'compareTrueParamsFunc' in params.keys():
    res['resComp'] = params['compareTrueParamsFunc'](dpmObjStd, res['std'])

  # print(res)

  return res