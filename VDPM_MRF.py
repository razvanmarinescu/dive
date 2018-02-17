import VDPMMean
import voxelDPM
import numpy as np
import scipy
import DisProgBuilder
import math
import gc
import sys

class VDPMMrfBuilder(VDPMMean.VDPMMeanBuilder):
  # builds a voxel-wise mean disease progression model with a Markov Random Field

  def __init__(self, isClust):
    super().__init__(isClust)

  def generate(self, dataIndices, expName, params):
    return VDPM_MRF(dataIndices, expName, params, self.plotterObj)

class VDPM_MRF(VDPMMean.VDPMMean):
  def __init__(self, dataIndices, expName, params, plotterObj):
    super().__init__(dataIndices, expName, params, plotterObj)

    # 5 is a good value for synthetic data, set to 0 for disabling MRF entirely
    self.alpha = self.params['alphaMRF'] # regularisation parameter for the MRF, clique potential is exp(a) or exp(-a)

  def recompResponsib(self, crossData, longData, crossAge1array, thetas, variances, subShiftsCross,
    trajFunc, prevClustProbBC, scanTimepts, crossPartCode, uniquePartCode):
    '''
    overwrite function as we need to include the MRF term. The function does the following:
     1. First estimates a MAP estimate Z^* over the cluster probabilities Z
        where (p(Z_l = k) - probability of vertex l belonging to cluster k). Perform this using
        Iterated conditional modes (ICM). Basically, it initialises Z to some value and then goes over
        each voxel and optimises its value given all the currently estimated values until convergence
     2. Computes a posterior distribution over each vertex assignment Z_l using the MAP values Z^* for
        the nodes neighbouring vertex l.

        p(Z_l|V_l, Theta^{old}, Z^*_{l_2 \neq l})

    '''


    clustProbParent, _, _ = super(VDPM_MRF, self).recompResponsib(crossData, longData, crossAge1array,
      thetas, variances, subShiftsCross, trajFunc, prevClustProbBC, scanTimepts, crossPartCode, uniquePartCode)

    clustProbTemp = clustProbParent
    nrIt = 5 # how many iterations to do between estimating clustProb and MRF
    for i in range(nrIt):
      logMrfTerms = self.estimMRFterms(clustProbTemp) # NR_VERTICEs x NR_CLUST
      logClustProbTemp = np.log(clustProbTemp) + logMrfTerms
      # renormalise the clust probabilities
      clustProbTemp = self.exponClustProb(logClustProbTemp)
      if np.isnan(logMrfTerms).any() or np.isnan(clustProbTemp).any():
        import pdb
        pdb.set_trace()
      assert not np.isnan(logMrfTerms).any()
      assert not np.isnan(clustProbTemp).any()

    # logClustProb = np.log(clustProbParent) + logMrfTerms
    # # renormalise the clust probabilities
    # clustProb = self.exponClustProb(logClustProb)

    clustProb = clustProbTemp

    # logMrfTerms = self.estimMRFterms(prevClustProbBC) # NR_VERTICEs x NR_CLUST
    # logClustProb = np.log(clustProbParent) + logMrfTerms
    # # renormalise the clust probabilities
    # clustProb = self.exponClustProb(logClustProb)

    # print('clustProb[:5,:]', clustProb[:5,:])
    # print('clustProbParent[:5,:]', clustProbParent[:5,:])
    # print(asda)

    return clustProb, crossData, longData

  def estimMRFterms(self, prevClustProbBC):
    nrBiomk, nrClust = prevClustProbBC.shape
    logMrfTerms = np.zeros((nrBiomk, nrClust), dtype=float)

    adjList = self.params['adjList']
    nrNeighVert = adjList.shape[1]
    assert adjList.shape[0] == nrBiomk
    expAmA = np.exp(self.alpha) - np.exp(-self.alpha)
    expmA = np.exp(-self.alpha)

    maxLikClust = np.argmax(prevClustProbBC, axis=1)

    pZlKAllBNC = np.zeros((nrBiomk,nrNeighVert,nrClust), dtype=float)

    for k in range(nrClust):
      for nv in range(nrNeighVert):
        # go over every neighbour node (generally 6 in FS)
        pZlK = prevClustProbBC[adjList[:,nv],k] #p(Z_l' = k) prob that neigh node has label k
        logMrfTerms[:,k] += np.log(expmA + expAmA * pZlK)

        pZlKAllBNC[:,nv,k] = pZlK

    pZlKAllBC = np.sum(pZlKAllBNC, axis=1)

    bmkWhichMatchInd = np.argmax(pZlKAllBC, axis=1) == maxLikClust
    bmkWhichDontMatchInd = np.logical_not(bmkWhichMatchInd)

    # print('logMrfTerms[bmkWhichDontMatchInd,:]',
    #   logMrfTerms[bmkWhichDontMatchInd,maxLikClust[bmkWhichDontMatchInd]])
    # print('logMrfTerms[bmkWhichMatchInd,:]',
    #   logMrfTerms[bmkWhichMatchInd,maxLikClust[bmkWhichMatchInd]])
    #
    # print('np.log(prevClustProbBC)', np.log(prevClustProbBC))
    # print('logMrfTerms', logMrfTerms)
    # print('prevClustProbBC', prevClustProbBC)
    # print(adsa)

    return logMrfTerms

  # def estimThetas(self, data, dpsCross, clustProbB, prevTheta, nrSubjLong):
  #   '''
  #   Optimise all 4 sigmoid parameters, don't fix the upper and lower bound anymore.
  #   :param data:
  #   :param dpsCross:
  #   :param clustProbB:
  #   :param prevTheta:
  #   :param nrSubjLong:
  #   :return:
  #   '''
  #
  #   dataWeightedS = np.sum(np.multiply(clustProbB[None, :], data), axis=1)
  #   objFuncLambda = lambda theta: self.objFunTheta(theta, dataWeightedS,
  #     dpsCross, clustProbB)[0]
  #
  #   if self.params['pdbPause']:
  #     import pdb
  #     pdb.set_trace()
  #
  #   # print('objFuncLambda([1,2,3,0])', objFuncLambda([1,2,3,0]))
  #   # print(adsas)
  #
  #   # objFuncDerivLambda = lambda theta: self.objFunThetaDeriv(theta, data, dpsCross, clustProbB)
  #
  #   # res = scipy.optimize.minimize(objFuncLambda, prevTheta, method='BFGS', jac=objFuncDerivLambda,
  #   #                               options={'gtol': 1e-8, 'disp': False})
  #
  #   initTheta = prevTheta
  #   res = scipy.optimize.minimize(objFuncLambda, initTheta, method='Nelder-Mead',
  #     options={'xtol': 1e-8, 'disp': True})
  #   bestTheta = res.x
  #
  #   # nrStartPoints = 10
  #   # nrParams = initTheta.shape[0]
  #   # pertSize = 1
  #   # minTheta = np.array([1, -np.inf, -np.inf, -np.inf])
  #   # maxTheta = np.array([np.inf, -1/np.std(dpsCross), np.inf, np.inf])
  #   # minSSD = np.inf
  #   # bestTheta = initTheta
  #   # for i in range(nrStartPoints):
  #   #   perturbTheta = initTheta * (np.ones(nrParams) + pertSize *
  #   #     np.random.multivariate_normal(np.zeros(nrParams), np.eye(nrParams)))
  #   #   # print('perturbTheta < minTheta', perturbTheta < minTheta)
  #   #   perturbTheta[perturbTheta < minTheta] = minTheta[perturbTheta < minTheta]
  #   #   perturbTheta[perturbTheta > maxTheta] = minTheta[perturbTheta > maxTheta]
  #   #   res = scipy.optimize.minimize(objFuncLambda, perturbTheta, method='Nelder-Mead',
  #   #     options={'xtol': 1e-8, 'disp': True})
  #   #   currTheta = res.x
  #   #   currSSD = res.fun
  #   #   print('currSSD', currSSD, objFuncLambda(currTheta))
  #   #   if currSSD < minSSD:
  #   #     minSSD = currSSD
  #   #     bestTheta = currTheta
  #   #     pertSize /= 1.2
  #   #   else:
  #   #     pertSize *= 1.2
  #   # print('bestTheta', bestTheta)
  #   # # print(adsa)
  #
  #
  #   newVariance = self.estimVariance(data, dpsCross, clustProbB, bestTheta, nrSubjLong)
  #
  #   return bestTheta, newVariance


  # def estimThetas(self, data, dpsCross, clustProbB, prevTheta, nrSubjLong):
  #
  #   recompThetaSig = lambda thetaFull, theta12: [thetaFull[0], theta12[0], theta12[1], thetaFull[3]]
  #
  #   dataWeightedS = np.sum(np.multiply(clustProbB[None, :], data), axis=1)
  #   objFuncLambda = lambda theta12: self.objFunTheta(recompThetaSig(prevTheta, theta12),
  #     dataWeightedS, dpsCross, clustProbB)[0]
  #
  #   # objFuncDerivLambda = lambda theta: self.objFunThetaDeriv(theta, data, dpsCross, clustProbB)
  #
  #   # res = scipy.optimize.minimize(objFuncLambda, prevTheta, method='BFGS', jac=objFuncDerivLambda,
  #   #                               options={'gtol': 1e-8, 'disp': False})
  #
  #   initTheta12 = prevTheta[[1, 2]]
  #   res = scipy.optimize.minimize(objFuncLambda, initTheta12, method='Nelder-Mead',
  #     options={'xtol': 1e-8, 'disp': True})
  #   bestTheta = res.x
  #
  #
  #   # nrStartPoints = 0
  #   # nrParams = initTheta12.shape[0]
  #   # pertSize = 1
  #   # minTheta = np.array([-1/np.std(dpsCross), -np.inf])
  #   # maxTheta = np.array([0, np.inf])
  #   # minSSD = np.inf
  #   # # bestTheta = initTheta12
  #   # success = False
  #   # for i in range(nrStartPoints):
  #   #   perturbTheta = initTheta12 * (np.ones(nrParams) + pertSize *
  #   #     np.random.multivariate_normal(np.zeros(nrParams), np.eye(nrParams)))
  #   #   # print('perturbTheta < minTheta', perturbTheta < minTheta)
  #   #   # perturbTheta[perturbTheta < minTheta] = minTheta[perturbTheta < minTheta]
  #   #   # perturbTheta[perturbTheta > maxTheta] = minTheta[perturbTheta > maxTheta]
  #   #   res = scipy.optimize.minimize(objFuncLambda, perturbTheta, method='Nelder-Mead',
  #   #     options={'xtol': 1e-8, 'disp': True})
  #   #   currTheta = res.x
  #   #   currSSD = res.fun
  #   #   print('currSSD', currSSD, objFuncLambda(currTheta))
  #   #   if currSSD < minSSD:
  #   #     # if we found a better solution then we decrease the step size
  #   #     minSSD = currSSD
  #   #     bestTheta = currTheta
  #   #     pertSize /= 1.2
  #   #     success = res.success
  #   #   else:
  #   #     # if we didn't find a solution then we increase the step size
  #   #     pertSize *= 1.2
  #   # print('bestTheta', bestTheta)
  #   # print(adsa)
  #
  #   # if not success:
  #   #   import pdb
  #   #   pdb.set_trace()
  #
  #   newTheta = recompThetaSig(prevTheta, bestTheta)
  #   #print(newTheta)
  #   newVariance = self.estimVariance(data, dpsCross, clustProbB, newTheta, nrSubjLong)
  #
  #   return newTheta, newVariance