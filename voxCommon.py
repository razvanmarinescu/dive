import os
import sys
from socket import gethostname
import numpy as np
import colorsys

demFullPath = os.path.abspath("../diffEqModel/")
voxFullPath = os.path.abspath("../")
sys.path.append(demFullPath)
sys.path.append(voxFullPath)

def addParserArgs(parser):

  parser.add_argument('--fwhmLevel', dest="fwhmLevel", type=int,default=0,
                      help='full-width half max level: 0, 5, 10, 15, 20 or 25')

  parser.add_argument('--runIndex', dest='runIndex', type=int,
                      default=1,help='index of run instance/process')

  parser.add_argument('--nrProc', dest='nrProc', type=int,
                     default=1,help='# of processes')

  parser.add_argument('--modelToRun', dest='modelToRun', type=int,
                     help='index of model to run')

  parser.add_argument('--models', dest='models',
                     help='index of first experiment to run')

  parser.add_argument('--nrOuterIt', dest='nrOuterIt', type=int,
                     help='# of outer iterations to run, for estimating clustering probabilities')

  parser.add_argument('--nrInnerIt', dest='nrInnerIt', type=int,
                     help='# of inner iterations to run, for fitting the model parameters and subj. shifts')

  parser.add_argument('--nrClust', dest='nrClust', type=int,
                     help='# of clusters to fit')

  parser.add_argument('--cluster', action="store_true", default=False,
                     help='need to include this flag if runnin on cluster')

  parser.add_argument('--initClustering', dest="initClustering", default='hist',
                     help='initial clustering method: k-means or hist')

  parser.add_argument('--agg', dest='agg', type=int, default=0,
                     help='agg=1 => plot figures without using Xwindows, for use on cluster where the plots cannot be displayed '
                    ' agg=0 => plot with Xwindows (for use on personal machine)')

  parser.add_argument('--rangeFactor', dest='rangeFactor', type=float,
                     help='factor x such that min -= rangeDiff*x/10 and max += rangeDiff*x/10')

  parser.add_argument('--informPrior', dest='informPrior', type=int, default=0,
                     help='enables informative prior based on gamma and gaussian dist')

  parser.add_argument('--alphaMRF', dest='alphaMRF', type=int, default=5,
                     help='alpha parameter for MRF, higher means more smoothing, lower means less. '
                          '5 was a good measure in simulations')

  parser.add_argument('--reduceSpace', dest='reduceSpace', type=int, default=1,
                     help='choose not to save certain files in order to reduce space')


def initCommonVoxParams(args):

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
    raise ValueError('check hostname or if running on cluster')

  nrClust = args.nrClust
  nrRows = int(np.sqrt(nrClust) * 0.95)
  nrCols = int(np.ceil(float(nrClust) / nrRows))
  assert (nrRows * nrCols >= nrClust)

  params = {}
  params['nrOuterIter'] = args.nrOuterIt
  params['nrInnerIter'] = args.nrInnerIt
  params['nrClust'] = nrClust
  params['runIndex'] = args.runIndex
  params['nrProcesses'] = args.nrProc
  params['rangeFactor'] = float(args.rangeFactor)
  params['cluster'] = args.cluster
  params['alphaMRF'] = args.alphaMRF # mrf alpha parameter, only used for MRF model
  params['initClustering'] = args.initClustering

  plotTrajParams = {}
  plotTrajParams['stagingHistNrBins'] = 20
  plotTrajParams['nrRows'] = nrRows
  plotTrajParams['nrCols'] = nrCols
  plotTrajParams['freesurfPath'] = freesurfPath
  plotTrajParams['blenderPath'] = blenderPath
  plotTrajParams['homeDir'] = homeDir
  plotTrajParams['reduceSpace'] = args.reduceSpace
  plotTrajParams['cluster'] = args.cluster
  plotTrajParams['TrajSamplesFontSize'] = 12
  plotTrajParams['TrajSamplesAdjBottomHeight'] = 0.175
  plotTrajParams['trajSamplesPlotLegend'] = True

  if args.agg:
    plotTrajParams['agg'] = True
  else:
    plotTrajParams['agg'] = False

  hostName = gethostname()
  if hostName == 'razvan-Inspiron-5547':
    height = 700
  else: #if hostName == 'razvan-Precision-T1700':
    height = 900

  width = 1300
  if nrClust <= 4:
    heightClust = height / 2
  elif 4 < nrClust <= 6:
    heightClust = int(height * 2/3)
    width = 900
  else:
    heightClust = height

  plotTrajParams['SubfigClustMaxWinSize'] = (width, heightClust)
  plotTrajParams['SubfigVisMaxWinSize'] = (width, height)

  plotTrajParams['clustHuePoints'] = np.linspace(0,1,nrClust,endpoint=False)
  plotTrajParams['clustCols'] = [colorsys.hsv_to_rgb(hue, 1, 1) for hue in plotTrajParams['clustHuePoints']]
  plotTrajParams['legendColsClust'] = min([nrClust, 4])

  return params, plotTrajParams