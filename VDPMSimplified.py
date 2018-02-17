import VDPM_MRF
import VDPMMean
import voxelDPM

class VDPMStaticBuilder(voxelDPM.VoxelDPMBuilder):
  # builds a voxel-wise disease progression model

  def __init__(self, isClust):
    super().__init__(isClust)

  def generate(self, dataIndices, expName, params):
    return VDPMStatic(dataIndices, expName, params, self.plotterObj)

class VDPMStatic(VDPMMean.VDPMMean):
  def __init__(self, dataIndices, expName, params, plotterObj):
    super().__init__(dataIndices, expName, params, plotterObj)

  def recompResponsib(self, crossData, longData, crossAge1array, thetas, variances,
    subShiftsCross, trajFunc, prevClustProbBC, scanTimepts, partCode, uniquePartCode):
    return prevClustProbBC, crossData, longData

class VDPMNoDPSBuilder(VDPM_MRF.VDPMMrfBuilder):
  # builds a voxel-wise disease progression model

  def __init__(self, isClust):
    super().__init__(isClust)

  def generate(self, dataIndices, expName, params):
    return VDPMNoDPS(dataIndices, expName, params, self.plotterObj)


class VDPMNoDPS(VDPM_MRF.VDPM_MRF):
  def __init__(self, dataIndices, expName, params, plotterObj):
    super().__init__(dataIndices, expName, params, plotterObj)

  # one option is to overwrite the run(), stageSubjects? and calcPredScores

  # the other option is to force subject shifts to be (1,0), effectively building the model against age.
  def estimShifts(self, dataOneSubj, thetas, variances, ageOneSubj1array, clustProbBC,
    prevSubShift, prevSubShiftAvg, fixSpeed):

    bestShift = [1, 0]

    return bestShift



