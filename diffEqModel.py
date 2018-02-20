from abc import ABC

import numpy as np
from scipy.interpolate import *
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, ConstantKernel, WhiteKernel)
from env import *
import copy
from matplotlib import pyplot as pl

def makeLongArray(array, scanTimepts, partCode):
  # place data in a longitudinal format
  longArray = [] # longArray can be data, diag, ageAtScan,scanTimepts, etc .. both 1D or 2D
  uniquePartCode = np.unique(partCode)
  nrParticipants = len(uniquePartCode)

  longCounter = 0

  for p in range(nrParticipants):
    currPartIndices = np.where(partCode == uniquePartCode[p])[0]
    currPartTimepoints = scanTimepts[currPartIndices]
    currPartTimeptsOrdInd = np.argsort(currPartTimepoints)
    # print uniquePartCode[p], currPartIndices, currPartTimepoints, currPartTimeptsOrdInd
    currPartIndicesOrd = currPartIndices[currPartTimeptsOrdInd]
    # print currPartIndicesOrd

    if len(currPartTimeptsOrdInd) > 1:
      longArray += [array[currPartIndicesOrd]]

  return longArray


def getCovMatFromNoise(estimNoiseZ):
  # noiseMean = np.nanmean(estimNoiseZ, axis=0)  # .reshape(-1,1)# average over all patients
  # print("noiseMean  ", noiseMean.shape, noiseMean)

  print(estimNoiseZ)
  covMat = np.diag(estimNoiseZ) ** 2

  return covMat

def getXshiftsFromNoise(estimNoiseZ, nrSamples):
    covMat = getCovMatFromNoise(estimNoiseZ)
    nrBiomk = estimNoiseZ.shape[0]
    print (covMat, np.zeros((nrBiomk,1)).reshape(-1,1).shape, covMat.shape)
    xValShifts = np.random.multivariate_normal(np.squeeze(np.zeros((nrBiomk,1))), covMat, nrSamples)
    return xValShifts

def createLongData2(data, diag, scanTimepts, partCode, ageAtScan):
  longData = makeLongArray(data, scanTimepts, partCode)
  longDiagAllTmpts = makeLongArray(diag, scanTimepts, partCode)
  longDiag = np.array([x[0] for x in longDiagAllTmpts])
  longScanTimepts = makeLongArray(scanTimepts, scanTimepts, partCode)
  longPartCode = makeLongArray(partCode, scanTimepts, partCode)
  longAgeAtScan = makeLongArray(ageAtScan, scanTimepts, partCode)

  return longData, longDiagAllTmpts, longDiag, longScanTimepts, longPartCode, longAgeAtScan


def createLongData(data, diag, scanTimepts, partCode, ageAtScan):

  # place data in a longitudinal format
  longData = []
  longDiagAllTmpts = []
  longDiag = []
  longScanTimepts = []
  longPartCode = []
  longAgeAtScan = []
  uniquePartCode = np.unique(partCode)
  nrParticipants = len(uniquePartCode)


  #longDiag = np.zeros(data.shape)
  #longScanTimepts = np.zeros(data.shape)
  longCounter = 0

  for p in range(nrParticipants):
    currPartIndices = np.where(partCode == uniquePartCode[p])[0]
    currPartTimepoints = scanTimepts[currPartIndices]
    currPartTimeptsOrdInd = np.argsort(currPartTimepoints)
    #print uniquePartCode[p], currPartIndices, currPartTimepoints, currPartTimeptsOrdInd
    currPartIndicesOrd = currPartIndices[currPartTimeptsOrdInd]
    #print currPartIndicesOrd
    
    if len(currPartTimeptsOrdInd) > 1:
      longData += [data[currPartIndicesOrd]]
      longDiagAllTmpts += [diag[currPartIndicesOrd]]
      longDiag += [diag[currPartIndicesOrd][0]]
      longScanTimepts += [scanTimepts[currPartIndicesOrd]]
      longPartCode += [uniquePartCode[p]]
      longAgeAtScan += [ageAtScan[currPartIndicesOrd]]

  return longData, longDiagAllTmpts, np.array(longDiag), longScanTimepts, longPartCode, longAgeAtScan

def calcDiffData(longData, longAgeAtScan):
  nrParticipants = len(longData)
  nrBiomk = longData[0].shape[1]
  dData = np.nan * np.ones((nrParticipants, nrBiomk), float)
  avgXdata = np.nan * np.ones((nrParticipants, nrBiomk), float)
  estimNoise = np.nan * np.ones((nrParticipants, nrBiomk), float)
  #fitIndices = [0,1,2] # only find linear fit using the first 2 timepoints
  for p in range(nrParticipants):
    #print longData[p].shape
    #print longAgeAtScan[p]
    assert(longData[p].shape[0] == longAgeAtScan[p].shape[0])
    assert(longData[p].shape[0] > 1)
    #if longData[p].shape[0] == 2:
    #  polyCoeff = np.polyfit(longAgeAtScan[p][[0,1]], longData[p][[0,1],:], deg=1) # shape(P, K), highest power first
    #else:  
    #polyCoeff = np.polyfit(longAgeAtScan[p][fitIndices], longData[p][fitIndices,:], deg=1) # shape(P, K), highest power first
    for b in range(nrBiomk):
      # do not include nan values
      fitIndices = np.logical_not(np.isnan(longData[p][:,b]))
      if np.sum(fitIndices) >= 2:
        polyCoeff = np.polyfit(longAgeAtScan[p][fitIndices], longData[p][fitIndices,b], deg=1) # shape(P, K),
        # highest power first

        # polyCoeff = np.polyfit(longAgeAtScan[p], longData[p], deg=1) # shape(P, K), highest power first
        #print "longAgeAtScan[p]", longAgeAtScan[p]
        #print polyCoeff.shape, polyCoeff
        dData[p,b] = polyCoeff[0]
        avgXdata[p,b] = np.nanmean(longData[p][:,b], axis=0)

      if np.sum(fitIndices) >= 4:
        estimNoise[p,b] = np.nanstd(longData[p][:,b], axis=0)

  estimNoise = np.nanmean(estimNoise,axis=0)
    #estimSquaredError[p,:] = 

  # assert not any(np.isnan(avgXdata))
  # assert not any(np.isnan(estimNoise))
  # assert not any(np.isnan(dData))

  return dData, avgXdata, estimNoise


def fitGaussianProc(patDXdTdata, patAvgXdata, dXdTdata, avgXdata, diag, lengthScaleFactors, plotTrajParams):
  '''
  Fits a GP on the change data (x, dx/dt)

  Parameters
  ----------
  patDXdTdata
  patAvgXdata
  dXdTdata
  avgXdata
  diag
  estimNoise
  lengthScaleFactors

  Returns
  -------

  '''

  # Mesh the input space for evaluations of the real function, the prediction and
  # its MSE
  assert(CTL == 1)
  nrBiomk = patDXdTdata.shape[1]
  #minX = np.amin(patAvgXdata, axis=0)
  #maxX = np.amax(patAvgXdata, axis=0)
  minX = np.nanmin(avgXdata, axis=0)
  maxX = np.nanmax(avgXdata, axis=0)
  assert not any(np.isnan(minX))
  assert not any(np.isnan(maxX))

  intervalSize = maxX-minX
  minX -= intervalSize/0.5
  maxX += intervalSize/0.5
  
  #print minX.shape, maxX.shape
  nrPointsToEval = 500
  x_pred = np.zeros((nrPointsToEval, nrBiomk),float)
  dXdT_pred = np.zeros((nrPointsToEval, nrBiomk),float)
  sigma_pred = np.zeros((nrPointsToEval, nrBiomk),float)
  nrSamples = 20
  posteriorSamples = np.zeros((nrSamples, nrPointsToEval, nrBiomk),float)

  # print(avgXdata.shape, diag.shape)
  # print(avgXdata[diag == CTL,:].shape)
  ctlXMean = np.nanmean(avgXdata[diag == CTL,:], axis = 0)
  ctlXStd = np.nanstd(avgXdata[diag == CTL,:], axis = 0)

  ctldXdTMean = np.nanmean(dXdTdata[diag == CTL,:], axis = 0)
  ctldXdTStd = np.nanstd(dXdTdata[diag == CTL,:], axis = 0)

  allXMean = np.nanmean(avgXdata, axis = 0)
  allXStd = np.nanstd(avgXdata, axis = 0)

  alldXdTMean = np.nanmean(dXdTdata, axis = 0)
  alldXdTStd = np.nanstd(dXdTdata, axis = 0)

  patXMean = np.nanmean(patAvgXdata, axis = 0)
  patXStd = np.nanstd(patAvgXdata, axis = 0)

  patdXdTMean = np.nanmean(patDXdTdata, axis = 0)
  patdXdTStd = np.nanstd(patDXdTdata, axis = 0)

  gpList = []
  
  for b in range(nrBiomk):
    points = np.linspace(minX[b], maxX[b], nrPointsToEval)
    #print points.shape

    X = patAvgXdata[:, b]
    Y = patDXdTdata[:, b]
    notNanInd = np.logical_not(np.isnan(X))
    X = X[notNanInd]
    Y = Y[notNanInd]

    X = X.reshape(-1,1)
    Y = Y.reshape(-1,1)

    # X = (X - allXMean[b]) / allXStd[b] # standardizing the inputs and outputs
    # Y = (Y - alldXdTMean[b]) / alldXdTStd[b]
    # minX[b] = (minX[b] - allXMean[b]) / allXStd[b]
    # maxX[b] = (maxX[b] - allXMean[b]) / allXStd[b]

    X = (X - patXMean[b]) / patXStd[b]  # standardizing the inputs and outputs
    # Y = (Y - patdXdTMean[b]) / patdXdTStd[b]
    Y = Y  / patdXdTStd[b]
    minX[b] = (minX[b] - patXMean[b]) / patXStd[b]
    maxX[b] = (maxX[b] - patXMean[b]) / patXStd[b]

    #print 'Xshape, Yshape', X.shape, Y.shape
    lower, upper = np.abs(1/np.max(X)), np.abs(1/(np.min(X)+1e-6))
    if lower > upper:
      lower, upper = upper, lower
    mid = 1/np.abs(np.mean(X))

    # print("X", X[:20],'Y', Y[:20])
    # print(minX, maxX)

    #lengthScale = (np.max(X)-np.min(X))
    lengthScale = lengthScaleFactors[b] * (np.max(X) - np.min(X))/2
    estimNoise = np.var(Y)/2 # this should be variance, as it is placed as is on the diagonal of the kernel, which is a covariance matrix
    #estimAlpha = np.ravel((np.std(Y))**2)
    #estimAlpha = np.var(Y)/2
    estimAlpha = np.std(Y)*2
    boundsFactor = 2.0
    #estimAlpha = 0
    #need to specity bounds as the lengthScale is optimised in the fit
    rbfKernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=lengthScale, length_scale_bounds=(float(lengthScale)/boundsFactor, 1*lengthScale))
    whiteKernel = ConstantKernel(1.0, constant_value_bounds="fixed") * WhiteKernel(noise_level=estimNoise, noise_level_bounds=(float(estimNoise)/boundsFactor, boundsFactor*estimNoise))
    #rbfKernel = 1 * RBF(length_scale=lengthScale)
    #whiteKernel = 1 * WhiteKernel(noise_level=estimNoise)
    kernel = rbfKernel + whiteKernel
    #kernel = 1.0 * RBF(length_scale=lengthScale)
    print('\nbiomk %d  lengthScale %f  noise %f alpha %f'% (b, lengthScale, estimNoise, estimAlpha))
    #print estimAlpha.shape
    normalizeYflag = False
    #normalizeYflag = True

    gp = GaussianProcessRegressor(kernel=rbfKernel, alpha=estimAlpha, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=100, normalize_y=normalizeYflag)

    #gp = GaussianProcessRegressor(kernel=rbfKernel, alpha=estimAlpha, optimizer=None, n_restarts_optimizer=100, normalize_y=True)

    assert not any(np.isnan(X))
    assert not any(np.isnan(Y))
    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(X, Y)
    print("optimised kernel", gp.kernel_)#, "  theta", gp.kernel_.theta, " bounds", gp.kernel_.bounds)

    #gpNonOpt = GaussianProcessRegressor(kernel=rbfKernel, alpha=estimAlpha, optimizer=None, normalize_y=False)
    #gpNonOpt.fit(X,Y)
    #print("non-optimised kernel", gpNonOpt.kernel_)#,  "  theta", gpNonOpt.kernel_.theta, " bounds", gpNonOpt.kernel_.bounds)

    #gp = gpNonOpt

    # Make the prediction on the meshed x-axis (ask for Cov matrix as well)
    x_pred[:,b] = np.linspace(minX[b], maxX[b], nrPointsToEval)
    assert not any(np.isnan(x_pred[:,b]))
    dXdT_predCurr, cov_matrix = gp.predict(x_pred[:,b].reshape(-1,1), return_cov=True)

    # make sure dXdT is not too low, otherwise truncate the [minX, maxX] interval
    dXdTthresh = 1e-10
    tooLowMask = np.abs(np.ravel(dXdT_predCurr)) < dXdTthresh
    print(tooLowMask.shape)
    if np.sum(tooLowMask) > nrPointsToEval/10:
      print("Warning dXdT is too low, will restict the [minxX, maxX] interval")
      goodIndicesMask = np.logical_not(tooLowMask)
      #print(x_pred.shape, goodIndicesMask.shape)
      #print(x_pred[goodIndicesMask, b])
      minX[b] = min(x_pred[goodIndicesMask,b])
      maxX[b] = max(x_pred[goodIndicesMask,b])
      x_pred[:, b] = np.linspace(minX[b], maxX[b], nrPointsToEval)
      dXdT_predCurr, cov_matrix = gp.predict(x_pred[:,b].reshape(-1,1), return_cov=True)


    MSE = np.diagonal(cov_matrix)

    dXdT_pred[:,b] = np.ravel(dXdT_predCurr)
    sigma_pred[:,b] = np.ravel(np.sqrt(MSE))
    samples = gp.sample_y(x_pred[:,b].reshape(-1,1), n_samples=nrSamples, random_state=0)
    posteriorSamples[:,:,b] = np.squeeze(samples).T

    # renormalize the Xs and Ys
    # x_pred[:,b] = x_pred[:,b] * allXStd[b] + allXMean[b]
    # dXdT_pred[:,b] = dXdT_pred[:,b] * alldXdTStd[b] + alldXdTMean[b]
    # sigma_pred[:,b] = sigma_pred[:,b] * alldXdTStd[b]
    # posteriorSamples[:,:,b] = posteriorSamples[:,:,b]*alldXdTStd[b] + alldXdTMean[b]

    # renormalize the Xs and Ys
    # x_pred[:, b] = x_pred[:, b] * patXStd[b] + patXMean[b]
    # dXdT_pred[:, b] = dXdT_pred[:, b] * patdXdTStd[b] + patdXdTMean[b]
    # sigma_pred[:, b] = sigma_pred[:, b] * patdXdTStd[b]
    # posteriorSamples[:, :, b] = posteriorSamples[:, :, b] * patdXdTStd[b] + patdXdTMean[b]

    x_pred[:, b] = x_pred[:, b] * patXStd[b] + patXMean[b]
    dXdT_pred[:, b] = dXdT_pred[:, b] * patdXdTStd[b]
    sigma_pred[:, b] = sigma_pred[:, b] * patdXdTStd[b]
    posteriorSamples[:, :, b] = posteriorSamples[:, :, b] * patdXdTStd[b]

    # diagCol = plotTrajParams['diagColors']
    # fig = pl.figure(1)
    # nrDiags = np.unique(diag).shape[0]
    # for diagNr in range(1, nrDiags + 1):
    #   print(avgXdata.shape, diag.shape, dXdTdata.shape, diagCol, diagNr)
    #   pl.scatter(avgXdata[diag == diagNr, b], dXdTdata[diag == diagNr, b], color = diagCol[diagNr - 1])
    #
    # modelCol = 'r' # red
    # pl.plot(x_pred[:, b], dXdT_pred[:, b], '%s-' % modelCol, label = u'Prediction')
    # pl.fill(np.concatenate([x_pred[:, b], x_pred[::-1, b]]), np.concatenate(
    #   [dXdT_pred[:, b] - 1.9600 * sigma_pred[:, b], (dXdT_pred[:, b] + 1.9600 * sigma_pred[:, b])[::-1]]), alpha = .5,
    #         fc = modelCol, ec = 'None', label = '95% confidence interval')
    # for s in range(nrSamples):
    #   pl.plot(x_pred[:, b], posteriorSamples[s, :, b])
    # fig.show()

    params = gp.get_params(deep=True)
    #print 'kernel', gp.kernel
    #print 'params', params

    gpList.append(gp)

  #print(adsa)

  return x_pred, dXdT_pred, sigma_pred, gpList, posteriorSamples



def integrateTrajOne(xs,dXdT_pred):
  # convert input vectors xs and dXdT into column vectors (nrPoints,1)

  assert all(dXdT_pred < 0)

  dXdT_pred = np.matrix(dXdT_pred)
  xs = np.matrix(xs)
  if xs.shape[0] == 1:
    xs = xs.T
  if dXdT_pred.shape[0] == 1:
    dXdT_pred = dXdT_pred.T

  indices = np.array(range(len(xs)))
  dXs = (xs[indices] - xs[indices - 1]) 
  dXdivdXdT = np.divide(dXs, dXdT_pred[indices])
  t = np.cumsum(dXdivdXdT).T
  # print("integrateTraj", xs.shape, dXs.shape, dXdT_pred.shape, dXdivdXdT.shape, t.shape)
  #print t

  return t


def largestNonZeroCrossing(xPred, dXdTpred, overlapRange):
  """
  returns the X-points on which the sign of dXdT doesn't change

  Parameters
  ----------
  xPred - values of biomk
  dXdTpred - change in biomk value in 1y
  overlapRange - range on biomk values that needs to overlap with the resulting section

  Returns
  -------

  """

  if np.any(dXdTpred < 0) and np.any(dXdTpred > 0):
    xIndMin = 0
    zeroCrossSects = []
    oldZeroCrossFlag = dXdTpred[0] < 0
    sizeXSects = []
    isApproved = []
    isEnforcedDirList = []
    isWithinRangeList = []
    nrPoints = xPred.shape[0]
    xPredSect = np.zeros(xPred.shape)
    dXdTpredSect = np.zeros(xPred.shape)
    for i in range(1,nrPoints):
      newZeroCrossFlag = dXdTpred[i] < 0
      if newZeroCrossFlag != oldZeroCrossFlag: # then add the previous section
        zeroCrossSects.append((xIndMin, i-1))
        sizeXSects.append(xPred[i] - xPred[xIndMin])
        oldZeroCrossFlag = newZeroCrossFlag
        isEnforcedDir = dXdTpred[i-1] < 0
        assert(xPred[xIndMin] <= xPred[i-1])
        isWithinRange = ( xPred[xIndMin] < overlapRange[1] and overlapRange[0] < xPred[i-1] )
        isApproved.append(isEnforcedDir and isWithinRange)
        isEnforcedDirList.append(isEnforcedDir)
        isWithinRangeList.append(isWithinRange)

        #print((xPred[xIndMin], xPred[i]), overlapRange)
        xIndMin = i

    zeroCrossSects.append((xIndMin, nrPoints-1))
    sizeXSects.append(xPred[-1] - xPred[xIndMin])
    isEnforcedDir = dXdTpred[nrPoints-1] < 0
    isWithinRange = (xPred[xIndMin] < overlapRange[1] and overlapRange[0] < xPred[nrPoints-1])
    isApproved.append(isEnforcedDir and isWithinRange)
    isEnforcedDirList.append(isEnforcedDir)
    isWithinRangeList.append(isWithinRange)

    # keep only the sections that go towards the enfored direction

    # select largest section
    isApproved = np.array(isApproved)
    sizeXSects = np.array(sizeXSects)
    sizeXSectsFilt = np.zeros(sizeXSects.shape)
    sizeXSectsFilt[isApproved] = sizeXSects[isApproved]
    #maxIndFilt = np.argmax(np.array(sizeXSectsFilt))
    maxInd = np.argmax(np.array(sizeXSectsFilt))

    #print("dXdTpred", dXdTpred)
    #assert(maxInd == maxIndFilt)

    if not any(isApproved):
      print("Sects", sizeXSects, zeroCrossSects)
      print(isApproved, isEnforcedDirList, isWithinRangeList)
      print(sizeXSects, sizeXSectsFilt)
      print(maxInd, zeroCrossSects[maxInd])
      raise AssertionError("need to have at least one interval approved")

    #xPredSect = xPred[zeroCrossSects[maxInd][0]:zeroCrossSects[maxInd][1]]
    #dXdTpredSect = dXdTpred[zeroCrossSects[maxInd][0]:zeroCrossSects[maxInd][1]]
    #xPredSect = np.linspace(xPred[zeroCrossSects[maxInd][0]], xPred[zeroCrossSects[maxInd][1]],nrPoints)
    indices = zeroCrossSects[maxInd]
    # indices = [indices[0]+3, indices[1]-3]
  else:
    #xPredSect = xPred
    indices = [0, xPred.shape[0]-1]
    #dXdTpredSect = dXdTpred

  # print(dXdTpred[indices[0]:indices[1]])
  assert all(dXdTpred[indices[0]:indices[1]] < 0)
  # print(asdsa)

  #print xPredSect.shape, dXdTpredSect.shape

  return indices


def integrateTrajAll(x_pred, dXdT_pred, avgXdata):
  nrPoints = x_pred.shape[0]
  nrBiomk = x_pred.shape[1]
  tsNzSect = np.zeros((nrPoints, nrBiomk),float)
  xPredNzSect = np.zeros((nrPoints, nrBiomk),float)
  dXdTpredNzSect = np.zeros((nrPoints, nrBiomk),float)
  badSamples = np.zeros(nrBiomk, bool)
  minX = np.nanmin(avgXdata, axis=0)
  maxX = np.nanmax(avgXdata, axis=0)

  success = True
  biomkFailList = []

  for b in range(nrBiomk):
    #print "biomk ", b
    #tmp = integrateTraj(x_pred[:,b],dXdT_pred[:,b])
    # make sure there are no zero crossings, otherwise remove them
    try:
      indicesNZ = largestNonZeroCrossing(x_pred[:,b], dXdT_pred[:,b], (minX[b], maxX[b]))

      # if b == 13:
      #   print(indicesNZ)
      #   print(dXdT_pred[indicesNZ[0],b], dXdT_pred[indicesNZ[1],b])
        # print(asda)

      xPredNzSect[:,b] = np.linspace(x_pred[indicesNZ[0],b], x_pred[indicesNZ[1],b], nrPoints)
      xPredTmp = x_pred[indicesNZ[0]:indicesNZ[1]+1,b]
      dXdTpredTmp = dXdT_pred[indicesNZ[0]:indicesNZ[1]+1,b]

      #f = interp1d(xPredTmp, dXdTpredTmp, bounds_error=False, fill_value="extrapolate")
      # print(b,indicesNZ)
      # print('xPredTmp', xPredTmp, 'dXdTpredTmp', dXdTpredTmp, 'x_pred', x_pred, 'dXdT_pred', dXdT_pred)
      # f = UnivariateSpline(xPredTmp, dXdTpredTmp, k=1, s=0)
      f = InterpolatedUnivariateSpline(xPredTmp, dXdTpredTmp, k = 1)
      #print [min(xPredNzSect[:,b]), max(xPredNzSect[:,b])], [min(xPredTmp), max(xPredTmp)]
      dXdTpredNzSect[:,b] = f(xPredNzSect[:,b])
      dXdTpredNzSect[dXdTpredNzSect[:,b] > 0,b] = -0.1**10
      # print(ads)

      # print('dXdT_pred', dXdTpredTmp)
      # print('dXdTpredNzSect after interp', dXdTpredNzSect[:, b])
      # print(x_pred[indicesNZ[0],b], x_pred[indicesNZ[1],b])
      # print('xPredNzSect[:,b]', xPredNzSect[:,b])
      # print('xPredTmp', xPredTmp)
      if not all(dXdTpredNzSect[:,b] < 0):
        raise ValueError('biomk %d diff model is positive!, need to flip increasing ' % b +
                         'biomk or there is not enough signal')

      tsTmp = np.ravel(integrateTrajOne(xPredNzSect[:,b],dXdTpredNzSect[:,b]))

      #print "ts[:,b]",ts[:,b]
      tsTmp = tsTmp - tsTmp[0] # make the trajectory start from zero
      tsNzSect[:,b] = tsTmp

    except (AssertionError):
      print("No feasible interval found for biomk ", b)
      success = False
      biomkFailList += [b]
      badSamples[b] = True

  return xPredNzSect, dXdTpredNzSect, tsNzSect, badSamples, biomkFailList, success

def filterDDSPA(params, excludeIDlocal):
  # create data folds
  filterIndices = np.logical_not(np.in1d(params['diag'], excludeIDlocal))
  filteredParams = copy.deepcopy(params)
  filteredParams['data'] = params['data'][filterIndices,:]
  filteredParams['diag'] = params['diag'][filterIndices]
  filteredParams['scanTimepts'] = params['scanTimepts'][filterIndices]
  filteredParams['partCode'] = params['partCode'][filterIndices]
  filteredParams['ageAtScan'] = params['ageAtScan'][filterIndices]

  return filteredParams

def filterDDSPAIndices(params, filterIndices):
  # create data folds
  filteredParams = copy.deepcopy(params)
  filteredParams['data'] = params['data'][filterIndices,:]
  filteredParams['diag'] = params['diag'][filterIndices]
  filteredParams['scanTimepts'] = params['scanTimepts'][filterIndices]
  filteredParams['partCode'] = params['partCode'][filterIndices]
  filteredParams['ageAtScan'] = params['ageAtScan'][filterIndices]

  return filteredParams


def filterDDSPAIndicesShallow(params, filterIndices):
  # make a shallow copy instead, slicing should make a shallow copy in python
  filteredParams = params
  filteredParams['data'] = params['data'][filterIndices,:]
  filteredParams['diag'] = params['diag'][filterIndices]
  filteredParams['scanTimepts'] = params['scanTimepts'][filterIndices]
  filteredParams['partCode'] = params['partCode'][filterIndices]
  filteredParams['ageAtScan'] = params['ageAtScan'][filterIndices]

  return filteredParams