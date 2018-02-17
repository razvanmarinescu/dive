import voxelDPM
# import VDPMMean
import numpy as np
import scipy
import sys
from env import *
import os
import pickle
import sklearn
import math
import numpy.ma as ma
import VDPMNan


''' Class for a Voxelwise Disease Progression Model that can handle missing data (NaNs).
    uses masked arrays for fitting the model (hence no data inference in E-step)'''

class VDPMNanMasksBuilder(voxelDPM.VoxelDPMBuilder):
  # builds a voxel-wise disease progression model

  def __init__(self, isClust):
    super().__init__(isClust)

  def generate(self, dataIndices, expName, params):
    return VDPMNanMasks(dataIndices, expName, params, self.plotterObj)

class VDPMNanMasks(VDPMNan.VDPMNan):
  def __init__(self, dataIndices, expName, params, plotterObj):
    super().__init__(dataIndices, expName, params, plotterObj)

    self.nanMask = np.nan


  def runInitClust(self, runPart, crossData, crossDiag):
    # printStats(longData, longDiag)
    nrClust = self.params['nrClust']

    os.system('mkdir -p %s' % self.outFolder)
    initClust = np.array(range(nrClust))
    assert nrClust == crossData.shape[1]

    return initClust

  def inferMissingData(self, crossData,longData, prevClustProbBC, thetas, subShiftsCross,
    crossAge1array, trajFunc, scanTimepts, partCode, uniquePartCode, plotterObj):

    ''' don't do anything, leave data with NaNs in this model! '''

    self.nanMask = np.isnan(crossData)
    plotterObj.nanMask = self.nanMask
    plotterObj.longDataNaNs = longData

    crossDataMasked = np.ma.masked_array(crossData, np.isnan(crossData))
    longDataMasked = [np.ma.masked_array(d, np.isnan(d))
      for d in longData]

    return crossDataMasked, longDataMasked

  def recompResponsib(self, crossData, longData, crossAge1array, thetas, variances, subShiftsCross,
    trajFunc, prevClustProbBC, scanTimepts, partCode, uniquePartCode):
    # overwrite function as we need to use a different variance (in the biomk measurements as opposed to their mean)



    return prevClustProbBC, crossData, longData



  def estimShifts(self, dataOneSubjTB, thetas, variances, ageOneSubj1array, clustProbBC,
    prevSubShift, prevSubShiftAvg, fixSpeed):

    '''
    do not use dot product because when NaNs are involved the weights will not sum to 1.
    use np.ma.average(.., weights) instead, as the weights will be re-normalised accordingly
    '''

    clustProbBCColNorm = clustProbBC / np.sum(clustProbBC, 0)[None, :]

    nrBiomk, nrClust = clustProbBC.shape
    nrTimepts = dataOneSubjTB.shape[0]

    dataOneSubjBT = dataOneSubjTB.T

    # declare it as masked array, compute it for every cluster with ma.average
    dataOneSubjWeightedCT = ma.zeros((nrClust, nrTimepts), float)
    for c in range(nrClust):
      dataOneSubjWeightedCT[c,:] = ma.average(dataOneSubjBT, axis=0, weights=clustProbBCColNorm[:, c])

    # convert back to np array for speed, do the calculation manually.
    dataOneSubjWeiManMaskCT = np.array(dataOneSubjWeightedCT)
    dataOneSubjWeiManMaskCT[dataOneSubjWeightedCT.mask] = np.nan


    if fixSpeed: # fixes parameter alpha to 1
      composeShift = lambda beta: [prevSubShiftAvg[0], beta]
      initSubShift = prevSubShift[1]
      # objFuncLambda2 = lambda beta: self.objFunShift(composeShift(beta), dataOneSubjWeightedCT, thetas,
      #   variances, ageOneSubj1array, clustProbBC)
      objFuncLambda = lambda beta: self.objFunShiftMaskedManual(composeShift(beta), dataOneSubjWeiManMaskCT, thetas,
        variances, ageOneSubj1array, clustProbBC)

      prevSubShiftAvgCurr = prevSubShiftAvg[1].reshape(1,-1)
    else:
      composeShift = lambda shift: shift
      initSubShift = prevSubShift
      # objFuncLambda2 = lambda shift: self.objFunShift(shift, dataOneSubjWeightedCT, thetas,
      #   variances, ageOneSubj1array, clustProbBC)
      objFuncLambda = lambda beta: self.objFunShiftMaskedManual(composeShift(beta), dataOneSubjWeiManMaskCT, thetas,
        variances, ageOneSubj1array, clustProbBC)

      prevSubShiftAvgCurr = prevSubShiftAvg

    # assert objFuncLambda(initSubShift) == objFuncLambda2(initSubShift)
    # print(adsa)

    # print('objFuncLambda(initSubShift)', objFuncLambda(initSubShift))

    res = scipy.optimize.minimize(objFuncLambda, initSubShift, method='Nelder-Mead',
                                  options={'xatol': 1e-2, 'disp': False})
    bestShift = res.x
    nrStartPoints = 2
    nrParams = prevSubShiftAvgCurr.shape[0]
    pertSize = 1
    minSSD = res.fun
    success = False
    for i in range(nrStartPoints):
      perturbShift = prevSubShiftAvgCurr * (np.ones(nrParams) + pertSize *
        np.random.multivariate_normal(np.zeros(nrParams), np.eye(nrParams)))
      res = scipy.optimize.minimize(objFuncLambda, perturbShift, method='Nelder-Mead',
        options={'xtol': 1e-8, 'disp': False, 'maxiter': 100})
      currShift = res.x
      currSSD = res.fun
      # print('currSSD', currSSD, objFuncLambda(currShift))
      if currSSD < minSSD:
        # if we found a better solution then we decrease the step size
        minSSD = currSSD
        bestShift = currShift
        pertSize /= 1.2
        success = res.success
      else:
        # if we didn't find a solution then we increase the step size
        pertSize *= 1.2
    print('bestShift', bestShift)

    return composeShift(bestShift)

  def objFunShift(self, shift, dataOneSubjWeightedCT, thetas, variances,
                  ageOneSubj1array, clustProbBC):



    # print('dataOneSubjWeightedCT', dataOneSubjWeightedCT.dtype)
    # print('ageOneSubj1array', ageOneSubj1array.dtype)
    # print('clustProbBC', clustProbBC.dtype)
    # print(adsas)

    dps = np.sum(np.multiply(shift, ageOneSubj1array), 1)
    nrClust = thetas.shape[0]
    # for tp in range(dataOneSubj.shape[0]):
    sumSSD = 0
    gammaInvK = np.sum(clustProbBC, 0)
    # print('dps', dps)
    sqErrorsK = ma.zeros(nrClust)
    for k in range(nrClust):
      sqErrorsK[k] = ma.sum(ma.power(dataOneSubjWeightedCT[k,:] -
                                  self.trajFunc(dps, thetas[k, :]), 2))

    sumSSD = ma.sum((sqErrorsK * gammaInvK)/ (2 * variances))


    assert not isinstance(sumSSD, ma.MaskedArray)

    # print('SqError', sqErrorsK)
    # print('gammaInvK', gammaInvK)
    # print('variances', variances)
    # print('sumSSD', sumSSD)

    logPriorShift = self.logPriorShiftFunc(shift, self.paramsPriorShift)

    # print('logPriorShift', logPriorShift, 'sumSSD', sumSSD)
    # print(sumSSD)
    # print(adsdsa)
    # if shift[0] < -400: # and -67
    #   import pdb
    #   pdb.set_trace()

    return sumSSD - logPriorShift

  def objFunShiftMaskedManual(self, shift, dataOneSubjWeightedCT, thetas, variances,
                  ageOneSubj1array, clustProbBC):

    # print('dataOneSubjWeightedCT', dataOneSubjWeightedCT.dtype)
    # print('ageOneSubj1array', ageOneSubj1array.dtype)
    # print('clustProbBC', clustProbBC.dtype)
    # print(adsas)

    dps = np.sum(np.multiply(shift, ageOneSubj1array), 1)
    nrClust = thetas.shape[0]
    # for tp in range(dataOneSubj.shape[0]):
    sumSSD = 0
    gammaInvK = np.sum(clustProbBC, 0)
    # print('dps', dps)
    sqErrorsK = np.zeros(nrClust)
    for k in range(nrClust):
      sqErrorsK[k] = np.nansum(np.power(dataOneSubjWeightedCT[k,:] -
                                  self.trajFunc(dps, thetas[k, :]), 2))

    sumSSD = np.nansum((sqErrorsK * gammaInvK)/ (2 * variances))



    # print('SqError', sqErrorsK)
    # print('gammaInvK', gammaInvK)
    # print('variances', variances)
    # print('sumSSD', sumSSD)

    logPriorShift = self.logPriorShiftFunc(shift, self.paramsPriorShift)

    # print('logPriorShift', logPriorShift, 'sumSSD', sumSSD)
    # print(sumSSD)
    # print(adsdsa)
    # if shift[0] < -400: # and -67
    #   import pdb
    #   pdb.set_trace()

    return sumSSD - logPriorShift


  def estimThetas(self, data, dpsCross, clustProbB, prevTheta, nrSubjLong):
    '''
    data contains NaNs.
    '''

    recompThetaSig = lambda thetaFull, theta12: [thetaFull[0], theta12[0], theta12[1], thetaFull[3]]

    # use masked arrays, clustProbB as the weights which get automatically normalised
    # depending on NaNs in the row.
    dataWeightedS = ma.average(data, axis=1, weights=clustProbB)
    objFuncLambda = lambda theta12: self.objFunTheta(recompThetaSig(prevTheta, theta12),
      dataWeightedS, dpsCross, clustProbB)[0]

    # objFuncDerivLambda = lambda theta: self.objFunThetaDeriv(theta, data, dpsCross, clustProbB)

    # res = scipy.optimize.minimize(objFuncLambda, prevTheta, method='BFGS', jac=objFuncDerivLambda,
    #                               options={'gtol': 1e-8, 'disp': False})

    initTheta12 = prevTheta[[1, 2]]

    nrStartPoints = 10
    nrParams = initTheta12.shape[0]
    pertSize = 1
    minTheta = np.array([-1/np.std(dpsCross), -np.inf])
    maxTheta = np.array([0, np.inf])
    minSSD = np.inf
    bestTheta = initTheta12
    success = False
    for i in range(nrStartPoints):
      perturbTheta = initTheta12 * (np.ones(nrParams) + pertSize *
        np.random.multivariate_normal(np.zeros(nrParams), np.eye(nrParams)))
      # print('perturbTheta < minTheta', perturbTheta < minTheta)
      # perturbTheta[perturbTheta < minTheta] = minTheta[perturbTheta < minTheta]
      # perturbTheta[perturbTheta > maxTheta] = minTheta[perturbTheta > maxTheta]
      res = scipy.optimize.minimize(objFuncLambda, perturbTheta, method='Nelder-Mead',
        options={'xtol': 1e-8, 'disp': True, 'maxiter':100})
      currTheta = res.x
      currSSD = res.fun
      print('currSSD', currSSD, objFuncLambda(currTheta))
      if currSSD < minSSD:
        # if we found a better solution then we decrease the step size
        minSSD = currSSD
        bestTheta = currTheta
        pertSize /= 1.2
        success = res.success
      else:
        # if we didn't find a solution then we increase the step size
        pertSize *= 1.2
    print('bestTheta', bestTheta)
    # print(adsa)

    # if not success:
    #   import pdb
    #   pdb.set_trace()

    newTheta = recompThetaSig(prevTheta, bestTheta)
    #print(newTheta)
    newVariance = self.estimVariance(data, dpsCross, clustProbB, newTheta, nrSubjLong)

    return newTheta, newVariance


  def objFunTheta(self, theta, dataWeightedS, dpsCross, _):

    sqErrorsS = ma.power((dataWeightedS - self.trajFunc(dpsCross, theta)), 2)
    meanSSD = ma.sum(sqErrorsS)

    assert not isinstance(meanSSD, ma.MaskedArray)

    logPriorTheta = self.logPriorThetaFunc(theta, self.paramsPriorTheta)

    return meanSSD - logPriorTheta, meanSSD


