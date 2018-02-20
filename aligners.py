from abc import ABC
import numpy as np
from scipy import interpolate
import scipy
from plotFunc import *
from aux import *
from env import *

class Aligner(ABC):
  def alignTrajXVal(self, ts, xsZ, xToAlign):
    """
     aligns trajectories to a fixed std dev level away from controls
     assumes xsZ are z-scores of x-values
    Parameters
    ----------
    ts - vector of timepoints
    xsZ - vector of biomarker values for each timepoint
    xToAlign - (nrBiomk,1) array of biomarker values to align with

    Returns
    -------
    tsNew - shifted trajectories
    xToAlignIsOutside - nrTraj array of flags specifying if the x used to align the traj is outside

    """
    nrPoints = xsZ.shape[0]
    nrBiomk = xsZ.shape[1]

    tsNew = np.zeros(ts.shape)
    xToAlignIsOutside = np.zeros(nrBiomk, bool)
    for b in range(nrBiomk):
      closestP = 0 # find which x-value is closest to the stdLevel
      minDist = np.abs(xsZ[0 ,b] - xToAlign[b])
      for p in range(nrPoints):
        newDist = np.abs(xsZ[p ,b] - xToAlign[b])
        if newDist < minDist:
          closestP = p
          minDist = newDist

      tsNew[: ,b] = ts[: ,b] - ts[closestP ,b]

      xToAlignIsOutside[b] = xToAlign[b] < min(xsZ[: ,b]) or xToAlign[b] > max(xsZ[: ,b])

    return tsNew, xToAlignIsOutside, xToAlign

class AlignerBaseVisit(Aligner):


  def align(self, dpmObj, tsNzSect, xsZ, longData, longDiag):
    anchorID = dpmObj.params['anchorID']
    muCtlData = dpmObj.muCtlData
    sigmaCtlData = dpmObj.sigmaCtlData
    blDataPatMean = calcBlDataPatMeanZ(tsNzSect, xsZ, longData, longDiag,
                                       anchorID, muCtlData, sigmaCtlData)
    return self.alignTrajXVal(tsNzSect, xsZ, blDataPatMean)


class AlignerBaseVisitNoise(AlignerBaseVisit):

  # xValShift

  # def __init__(self, xValShift):
  #  self.xValShift = xValShift

  # overwrite parent's method
  def alignNoise(self, tsNzSect, xsZ, longData, longDiag, anchorID, muCtlData, sigmaCtlData, xValShift):
    # aligns traj to biomk mean value at baseline visit of patients, but adds gaussian noise to the alignment
    blDataPatMean = calcBlDataPatMeanZ(tsNzSect, xsZ, longData, longDiag,
                                       anchorID, muCtlData, sigmaCtlData)
    return self.alignTrajXVal(tsNzSect, xsZ, blDataPatMean + xValShift)

def calcBlDataPatMeanZ(tsNzSect, xsZ, longData, longDiag, anchorID, muCtlData, sigmaCtlData):
  # estimate avg value at baseline visit of patients for each biomk
  nrSubj = len(longData)
  nrBiomk = longData[0].shape[1]
  blData = np.zeros((nrSubj, nrBiomk))
  for i in range(len(longData)):
    blData[i ,:] = longData[i][0 ,:]

  blDataPat = blData[longDiag == anchorID, :]
  blDataPatMean = np.nanmean(blDataPat, axis=0)
  blDataPatMeanZ = (blDataPatMean - muCtlData) / sigmaCtlData # convert to z-score

  print(blDataPatMean.shape)
  print(blDataPatMean)
  print(np.nanmax(xsZ), np.nanmin(xsZ))
  assert(len(blDataPatMeanZ) == nrBiomk)

  # align traj to the mean of the baseline visit of patients

  return blDataPatMeanZ

class AlignerEM(Aligner):

  def align(self, dpmObj, tsNzSect, xsZ, longData, longDiag):
    anchorID = dpmObj.params['anchorID']
    muCtlData = dpmObj.muCtlData
    sigmaCtlData = dpmObj.sigmaCtlData
    # aligns traj to biomk mean value at baseline visit of patients, but adds gaussian noise to the alignment
    blDataPatMean = calcBlDataPatMeanZ(tsNzSect, xsZ, longData, longDiag,
                                       anchorID, muCtlData, sigmaCtlData)
    tsPatMeanAligned, xToAlignIsOutside = self.alignTrajXVal(tsNzSect, xsZ, blDataPatMean)
    test = tsPatMeanAligned[0,1]
    dpmObj.ts = tsPatMeanAligned

    patMask = np.logical_not(np.in1d(dpmObj.diag, dpmObj.params['excludeXvalidID']))
    dataNonZ = dpmObj.data[patMask ,:]
    data = dpmObj.getDataZ(dataNonZ) # convert data to z-scores as this is how the traj work
    diag = dpmObj.diag[patMask ]
    nrSubj, nrBiomk = dataNonZ.shape
    nrPoints = tsPatMeanAligned.shape[0]

    initBiomkShifts = np.zeros(nrBiomk)
    initSubjStages = np.zeros(nrSubj)

    minBiomkS = -20
    maxBiomkS = 20

    nrIterations = 10
    nrBiomkShifts = 100
    biomkShifts = np.zeros((nrBiomk,nrIterations))
    sigmaSqs = np.zeros((nrBiomk,nrIterations))
    sigmaSqs[:, 0] = np.power(np.mean(dpmObj.estimNoiseZ, axis=0),2)
    maxLikStages = np.zeros((nrSubj,nrIterations))

    patStagesSet = dpmObj.params['tsStages']
    biomkShiftsSet = np.linspace(minBiomkS, maxBiomkS, num=nrBiomkShifts)

    dataIndicesStaging = np.arange(0, len(dpmObj.diag), 1)

    fs = [interpolate.interp1d(tsPatMeanAligned[:, b], xsZ[:, b], kind='linear', fill_value='extrapolate')
          for b in range(nrBiomk)]

    logL = np.zeros(nrIterations)  # usually is -inf because of the biomk shifts normalisation (sum to 0)
    logL[0] = calcIncompleteDataLogL(data, fs, patStagesSet, biomkShifts[:, 0],
                                     sigmaSqs[:, 0])

    print('itNr %d ' % 0, 'logL', logL[0])

    for itNr in range(1,nrIterations):
      # E-step - estimate the staging probabilities of subjects given the current alignment of traj
      # i.e. E_{p(Z|X,theta_old)}
      dpmObj.ts = tsPatMeanAligned + np.tile(biomkShifts[:,itNr-1], (nrPoints, 1))
      dpmObj.covMatNoiseZ = np.diag(sigmaSqs[:, itNr-1]) # already contains sigma squares, just put in diag matrix
      (maxLikStages[:, itNr], maxStagesIndex, stagingProb, stagingLik, _) = \
        dpmObj.stageSubjectsData(dataNonZ)

      # M-step - estimate the traj alignment by maximising E_{p(Z|X,theta_old)} (log p(X,Z|theta))

      # estimate the optimal noise level sigma using the EM update
      sigmaSqs[:, itNr] = estimateSigmasEMupdate(stagingProb, data, fs, patStagesSet, biomkShifts[:,itNr])

      # now estimate the optimal biomk shifts given the stagingProb and the sigma levels
      QsumB = np.zeros((nrBiomk, nrBiomkShifts))
      for b in range(nrBiomk): # for each trajectory
        for bs, biomkShiftCurr in enumerate(biomkShiftsSet): # for each possible shift of that trajectory
          for s, stage in enumerate(patStagesSet): # for each possible stage of the subjects
            # try to vectorize over all the participants at that particular stage
            meanCurrStage = fs[b](stage-biomkShiftCurr)
            QsumB[b,bs] += np.sum(np.multiply(stagingProb[:,s], np.power(data[:,b] - meanCurrStage,2)))
            # print(stagingProb[:,s], np.power(dataZ[:,b] - meanCurrStage,2))
            # print(QsumB[b,bs])
            # print(asda)

        #find the biomk shift that caused the highest increase in Q(theta, theta^old)
        # take min as QsubB is a simplified version of Q(theta, theta^old) which needs to be minimized
        bsMin = np.argmin(QsumB[b,:])
        biomkShifts[b, itNr] = biomkShiftsSet[bsMin]
        # print(QsumB[b, :], bsMin, biomkShiftsSet[bsMin])
        # print(sdas)

      # print('maxLikStages[1:20]', maxLikStages[1:20])
      QsumMin = np.min(QsumB, axis=1)
      assert(QsumMin.shape[0] == nrBiomk)
      QsumSum = np.sum(QsumMin)
      Qtheta = calcQthetaThetaOld(stagingProb, data, fs, patStagesSet, biomkShifts[:, itNr],
                                   sigmaSqs[:, itNr])
      logL = calcIncompleteDataLogL(data, fs, patStagesSet, biomkShifts[:, itNr],
                                   sigmaSqs[:, itNr])

      print('itNr %d QsumSum %f ' % (itNr, QsumSum), 'Qtheta', Qtheta, 'logL', logL)
      print(' biomkShifts[:, itNr]',  biomkShifts[:, itNr])
      # print('QsumB[0,:]', QsumB[0,:])
      # print('QsumMin', QsumMin)
      # print('stagingProb[1:3,:]', stagingProb[1:3,:])
      # print(asdas)
      assert(tsPatMeanAligned[0,1] == test)
      # fig = plotTrajSubfigWithData(dpmObj.ts, dpmObj.xsZ, None, None, None, dpmObj.params['labels'],
      #    dpmObj.params['plotTrajParams'], dataZ, dpmObj.diag, maxLikStages[:, itNr], thresh=0)
      # fig.savefig('matfiles/%s/EMfig_it%d.png' % (dpmObj.expName, itNr), dpi=100)

    dpmObj.ts = tsPatMeanAligned + np.tile(biomkShifts[:, -1], (nrPoints, 1))
    print('Biomk shifts', biomkShifts[:,-1])
    # print(asdsa)
    res = {'biomkShifts' : biomkShifts}

    return dpmObj.ts, res

class AlignerLogLOpt(Aligner):

  def align(self, dpmObj, tsNzSect, xsZ, longData, longDiag):
    """
    aligns the trajectories using direct optimisation on the incomplete data logL from calcIncompleteDataLogL
    the incomplete data logL is the marginal of logL over the stages
    Parameters
    ----------
    dpmObj
    tsNzSect
    xsZ
    longData
    longDiag

    Returns
    -------

    """
    anchorID = dpmObj.params['anchorID']
    muCtlData = dpmObj.muCtlData
    sigmaCtlData = dpmObj.sigmaCtlData
    # aligns traj to biomk mean value at baseline visit of patients, but adds gaussian noise to the alignment
    blDataPatMean = calcBlDataPatMeanZ(tsNzSect, xsZ, longData, longDiag,
                                       anchorID, muCtlData, sigmaCtlData)
    tsPatMeanAligned, xToAlignIsOutside = self.alignTrajXVal(tsNzSect, xsZ, blDataPatMean)
    test = tsPatMeanAligned[0,1]
    dpmObj.ts = tsPatMeanAligned

    patMask = np.logical_not(np.in1d(dpmObj.diag, dpmObj.params['excludeXvalidID']))
    dataIndicesNN = np.logical_and(patMask, np.sum(np.isnan(dpmObj.data), 1) == 0)
    dataNonZ = dpmObj.data[dataIndicesNN ,:]
    data = dpmObj.getDataZ(dataNonZ) # convert data to z-scores as this is how the traj work
    diag = dpmObj.diag[dataIndicesNN]
    nrSubj, nrBiomk = dataNonZ.shape
    nrPoints = tsPatMeanAligned.shape[0]

    initBiomkShifts = np.zeros(nrBiomk)
    initSubjStages = np.zeros(nrSubj)

    minBiomkS = -20
    maxBiomkS = 20

    nrIterations = 30
    nrBiomkShifts = 100
    biomkShifts = np.zeros((nrBiomk,nrIterations))
    sigmaSqs = np.zeros((nrBiomk,nrIterations))
    sigmaSqs[:, 0] = np.power(np.mean(dpmObj.estimNoiseZ, axis=0),2)
    maxLikStages = np.zeros((nrSubj,nrIterations))

    patStagesSet = dpmObj.params['tsStages']
    biomkShiftsSet = np.linspace(minBiomkS, maxBiomkS, num=nrBiomkShifts)

    fs = [interpolate.interp1d(tsPatMeanAligned[:, b], xsZ[:, b], kind='linear', fill_value='extrapolate')
          for b in range(nrBiomk)]

    logL = np.zeros(nrIterations) # usually is -inf because of the biomk shifts normalisation (sum to 0)
    logL[0] = calcIncompleteDataLogL(data, fs, patStagesSet, biomkShifts[:, 0],
                                      sigmaSqs[:, 0])

    print('itNr %d ' % 0, 'logL', logL[0])

    for itNr in range(1,nrIterations):
      # estimate the optimal noise level sigma
      dpmObj.ts = tsPatMeanAligned + np.tile(biomkShifts[:,itNr-1], (nrPoints, 1))
      dpmObj.covMatNoiseZ = np.diag(sigmaSqs[:, itNr-1]) # already contains sigma squares, just put in diag matrix
      (maxLikStages[:, itNr], maxStagesIndex, stagingProb, stagingLik, _) = dpmObj.stageSubjectsData(dataNonZ)

      sigmaSqs[:, itNr] = estimateSigmasEMupdate(stagingProb, data, fs, patStagesSet, biomkShifts[:, itNr])

      # make sure the biomk shifts sum to zero, as there is one extra DOF
      addDOF = lambda x: np.append(x, 1-np.sum(x))
      fun = lambda x: -calcIncompleteDataLogL(data, fs, patStagesSet, addDOF(x), sigmaSqs[:, itNr])
      optRes = scipy.optimize.minimize(fun=fun, x0=biomkShifts[:-1,itNr-1], method='Powell')
      assert optRes.success
      print(optRes.x)
      biomkShifts[:, itNr] = addDOF(optRes.x)

      logL[itNr] = calcIncompleteDataLogL(data, fs, patStagesSet, biomkShifts[:, itNr],
                                   sigmaSqs[:, itNr])

      print('itNr %d ' % itNr, 'logL', logL[itNr])
      print('sigmaSqs[:, itNr]', sigmaSqs[:, itNr])
      print(' biomkShifts[:, itNr]',  biomkShifts[:, itNr])
      # print('QsumB[0,:]', QsumB[0,:])
      # print('QsumMin', QsumMin)
      # print('stagingProb[1:3,:]', stagingProb[1:3,:])
      # print(asdas)
      assert(tsPatMeanAligned[0,1] == test)
      # fig = plotTrajSubfigWithData(dpmObj.ts, dpmObj.xsZ, None, None, None, dpmObj.params['labels'],
      #    dpmObj.params['plotTrajParams'], dataZ, dpmObj.diag, maxLikStages[:, itNr], thresh=0)
      # fig.savefig('matfiles/%s/EMfig_it%d.png' % (dpmObj.expName, itNr), dpi=100)

      if logL[itNr] < logL[itNr - 1]:
        # break the loop as the logL is decreasing
        break

    print('Biomk shifts', biomkShifts[:,-1])
    # print(asdsa)

    # shift all traj with one constant so that 0-axis is optimally separating CTL from patients (or MCI)
    threshMask = np.in1d(diag, [CTL, dpmObj.params['anchorID']])
    # print(diag[threshMask], diag, dpmObj.params['anchorID'])
    if len(np.unique(diag[threshMask])) != 2:
      raise Exception("there should be exactly 2 groups for finding the ideal threshold")

    idealThresh = findIdealThresh(stagingProb[threshMask,:], diag[threshMask])
    dpmObj.ts -= patStagesSet[idealThresh] # center the traj around the ideal separating threshold

    res = {'biomkShifts' : biomkShifts, 'logL':logL}

    return dpmObj.ts, res

def estimateSigmasEMupdate(stagingProb, data, fs, patStagesSet, biomkShifts):
  nrSubj, nrBiomk = data.shape
  sigmaSqs = np.zeros(nrBiomk)
  for b in range(nrBiomk):
    for s, stage in enumerate(patStagesSet):  # for each possible stage of the subjects
      # try to vectorize over all the participants at that particular stage
      meanCurrStage = fs[b](stage - biomkShifts[b])
      sigmaSqs[b] += (1/nrSubj) * np.sum(np.multiply(stagingProb[:, s], np.power(data[:, b] - meanCurrStage, 2)))

  return sigmaSqs

def calcQthetaThetaOld(stagingProb, data, fs, patStagesSet, biomkShifts, sigmaSqs):
  nrSubj, nrBiomk = data.shape
  logLiks = np.zeros(stagingProb.shape)
  for s, stage in enumerate(patStagesSet):  # for each possible stage of the subjects
    # try to vectorize over all the participants at that particular stage
    meanCurrStage = [fs[b](stage - biomkShifts[b]) for b in range(nrBiomk)]
    logLiks[:,s] = scipy.stats.multivariate_normal.logpdf(data, meanCurrStage, np.diag(sigmaSqs))

  Qsum = np.sum(np.multiply(stagingProb, logLiks))
  # print(logLiks)
  # print(Qsum)
  # print(asdsa)

  return Qsum

def calcIncompleteDataLogL(data, fs, patStagesSet, biomkShifts, sigmaSqs):
  """
  computes the incomplete data log-likelihood, i.e. p(X|theta) = sum_Z p(X,Z|theta)
  so it sums ove all the possible stages of the patients

  Parameters
  ----------
  data
  fs
  patStagesSet
  biomkShifts
  sigmaSqs

  Returns
  -------

  """
  nrSubj, nrBiomk = data.shape
  nrStages = len(patStagesSet)
  liks = np.zeros((nrSubj, nrStages), float)
  for s, stage in enumerate(patStagesSet):  # for each possible stage of the subjects
    # try to vectorize over all the participants at that particular stage
    meanCurrStage = [fs[b](stage - biomkShifts[b]) for b in range(nrBiomk)]
    liks[:,s] = scipy.stats.multivariate_normal.pdf(data, meanCurrStage, np.diag(sigmaSqs))

  logL = np.sum(np.log(np.sum((1/nrStages) * liks, axis=1)))
  # print(liks)
  # print(logL)
  # print(asdsa)

  return logL