import os
import glob
import sys
import argparse
from socket import gethostname
from subprocess import check_output as qx
import subprocess
import nibabel as nib
import numpy as np
import csv
from datetime import datetime
import pickle


def addParserArgs(parser):
  parser.add_argument('--firstInstance', dest='firstInstance', type=int,
                      default=1,help='index of first run instance/process')

  parser.add_argument('--lastInstance', dest='lastInstance', type=int,
                     default=10,help='index of last run instance/process')

  parser.add_argument('--nrProc', dest='nrProc', type=int,
                     default=10,help='# of processes')

  parser.add_argument('--firstModel', dest='firstModel', type=int,
                     help='index of first experiment to run')

  parser.add_argument('--lastModel', dest='lastModel', type=int,
                     help='index of last experiment to run')

  parser.add_argument('--models', dest='models',
                     help='index of first experiment to run')

  parser.add_argument('--launcherScript', dest='launcherScript',default='adniThick.py',
                     help='launcher script, such as adniThick.py, etc ..')

  parser.add_argument('--cluster', action="store_true", help='set this if the program should run on the CMIC cluster')
  parser.add_argument('--timeLimit', dest='timeLimit',
                     help='timeLimit for the job in hours')
  parser.add_argument('--printOnly', action="store_true", help='only print experiment to be run, not actualy run it')

  # specific to the VDPM

  parser.add_argument('--initClustering', dest="initClustering", default='hist',
                     help='initial clustering method: k-means, hist or fsurf')

  parser.add_argument('--nrClust', dest='nrClust', type=int,
                     help='# of clusters to fit')

  parser.add_argument('--nrClustList', dest='nrClustList',
                     help='# of clusters to fit')

  parser.add_argument('--nrOuterIt', dest='nrOuterIt', type=int,
                     help='# of outer iterations to run, for estimating clustering probabilities')

  parser.add_argument('--nrInnerIt', dest='nrInnerIt', type=int,
                     help='# of inner iterations to run, for fitting the model parameters and subj. shifts')

  parser.add_argument('--mem', dest='mem', type=int,
                     help='memory limit of process')

  parser.add_argument('--agg', dest='agg', type=int, default=0,
                     help='plot figures without using Xwindows, for use on cluster, not linked to cluster as I need to test it locally first')

  parser.add_argument('--rangeFactor', dest='rangeFactor', type=float,
                     help='factor x such that min -= rangeDiff*x/10 and max += rangeDiff*x/10')

  parser.add_argument('--informPrior', dest='informPrior', type=int, default=0,
                     help='enables informative prior based on gamma and gaussian dist')

  parser.add_argument('--tscratch', dest='tscratch', type=int,
                     help='requests space on /scracth0 (GB)')

  parser.add_argument('--reserve', action="store_true",
                     help='uses resource reservation for making the job not queue for long')

  parser.add_argument('--serial', action="store_true",
                     help='runs jobs serially instead of spawning processes on local machine or on cluster')

  parser.add_argument('--alphaMRF', dest='alphaMRF', type=int, default=5,
    help='alpha parameter for MRF, higher means more smoothing, lower means less. '
         '5 was a good measure in simulations')


def initCommonLaunchParams(args):
  hostName = gethostname()
  print(hostName)

  launchParams = {}
  if hostName == 'razvan-Inspiron-5547':
    launchParams['freesurfPath'] = '/usr/local/freesurfer-5.3.0'
    launchParams['homeDir'] = '/home/razvan'
  elif hostName == 'razvan-Precision-T1700':
    launchParams['freesurfPath'] = '/usr/local/freesurfer-5.3.0'
    launchParams['homeDir'] = '/home/razvan'
  elif args.cluster:
    launchParams['freesurfPath'] = '/share/apps/freesurfer-5.3.0'
    launchParams['homeDir'] = '/home/rmarines'
  else:
    raise ValueError('hostname wrong or forgot --cluster flag')

  # MEM_LIMIT = 31.8 # in GB
  launchParams['MEM_LIMIT'] = 15.7 # in GB
  if args.mem:
    launchParams['MEM_LIMIT'] = args.mem

  launchParams['REPO_DIR'] = '%s/phd_proj/voxelwiseDPM' % launchParams['homeDir']
  launchParams['OUTPUT_DIR'] = '%s/clusterOutputVDPM' % launchParams['REPO_DIR']
  #launchParams['OUTPUT_DIR'] = '/cluster/project0/VWDPM/clusterOutput/'

  # exportSubjCmd = 'export SUBJECTS_DIR=%s' % SUBJECTS_DIR
  # exportFreeSurfCmd = 'export FREESURFER_HOME=%s; source %s/SetUpFreeSurfer.sh' % (freesurfPath,freesurfPath)

  if args.cluster:
    launchParams['WALL_TIME_LIMIT_HOURS'] = int(args.timeLimit)
    launchParams['WALL_TIME_LIMIT_MIN'] = 15

  if args.tscratch:
    launchParams['tscratchStr'] = ' -l tscratch=%dG ' % args.tscratch
  else:
    launchParams['tscratchStr'] = ''

  if args.reserve:
    launchParams['reserveStr'] = '-R y'
  else:
    launchParams['reserveStr'] = ''

  print('informPrior', args.informPrior)

  return launchParams

def getQsubCmdPart2(launchParams, jobName):
  qsubCmd = 'qsub -S /bin/bash -l tmem=%.1fG -l h_vmem=%.1fG -l ' \
            'h_rt=%d:%d:0 %s -N %s -j y -wd %s %s' % (launchParams['MEM_LIMIT'],
  launchParams['MEM_LIMIT'], launchParams['WALL_TIME_LIMIT_HOURS'],
  launchParams['WALL_TIME_LIMIT_MIN'],
  launchParams['tscratchStr'], jobName,
  launchParams['OUTPUT_DIR'], launchParams['reserveStr'])

  return qsubCmd
