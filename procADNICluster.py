import os
import glob
import sys
import argparse
from socket import gethostname
from subprocess import check_output as qx
import nibabel as nib
import numpy as np
import csv
from datetime import datetime
import pickle


parser = argparse.ArgumentParser(description='Launches freesurfer processes for ADNI data on cluster or local machine')

parser.add_argument('--cluster', action="store_true", help='set this if the program should run on the CMIC cluster')
parser.add_argument('--timeLimit', dest='timeLimit',
                   help='timeLimit for the job in hours')
parser.add_argument('--printOnly', action="store_true", help='only print experiment to be run, not actualy run it')

parser.add_argument('--step', dest="step", help='which step to run in the longitudinal pipeline')

parser.add_argument('--test', action="store_true", help='only for testing one subject')

args = parser.parse_args()

hostName = gethostname()
print(hostName)

if hostName == 'razvan-Inspiron-5547':
  homeDir = '/home/razvan'
  freesurfPath = '/usr/local/freesurfer-6.0.0'
elif hostName == 'razvan-Precision-T1700':
  homeDir = '/home/razvan'
  freesurfPath = '/usr/local/freesurfer-6.0.0'
elif args.cluster:
  homeDir = '/home/rmarines'
  freesurfPath = '/home/rmarines/src/freesurfer-6.0.0'
else:
  raise ValueError('wrong hostname or cluster flag')

# adniSubFd = 'MP-Rage_proc_all'
adniSubFd = 'ADNI2_MAYO'
SUBJECTS_DIR = '%s/ADNI_data/%s/subjects' % (homeDir, adniSubFd)
rawMriPath = '%s/ADNI_data/%s/ADNI' % (homeDir, adniSubFd)
 

MEM_LIMIT = 7.9 # in GB
REPO_DIR = '%s/phd_proj/voxelwiseDPM' % homeDir
OUTPUT_DIR = '%s/clusterOutputADNI' % REPO_DIR

exportSubjCmd = 'export SUBJECTS_DIR=%s' % SUBJECTS_DIR
exportFreeSurfCmd = 'export FREESURFER_HOME=%s; source %s/SetUpFreeSurfer.sh' % (freesurfPath,freesurfPath)

WALL_TIME_LIMIT_HOURS = int(args.timeLimit)
WALL_TIME_LIMIT_MIN = 15
step = int(args.step)

def getQsubCmd1(timept, t1_file, subjID):
  # if there's an error about tty, add & in the last parameter
  timeptID = "%s-%s" % (subjID, timept)
  jobName = 'A1_%s' % timeptID
  runCmd = '%s; %s;    %s/bin/recon-all -all -s %s -i %s' % (exportFreeSurfCmd, exportSubjCmd, freesurfPath, timeptID, t1_file)
  qsubCmd = 'qsub -S /bin/bash -l tmem=%.1fG -l h_vmem=%.1fG -l h_rt=%d:%d:0 -N %s -j y -wd %s ' % (MEM_LIMIT, MEM_LIMIT, WALL_TIME_LIMIT_HOURS, WALL_TIME_LIMIT_MIN, jobName, OUTPUT_DIR)

  cmdClust = 'echo "%s" | %s' % (runCmd, qsubCmd) # echo the matlab cmd then pipe it to qsub
  return cmdClust, runCmd

def getQsubCmd2(timeptList, subjID):
  templateID = "template_%s" % subjID
  jobName = 'A2_%s' % templateID
  runCmd = '%s; %s;    %s/bin/recon-all -base %s ' % (exportFreeSurfCmd, exportSubjCmd, freesurfPath, templateID)
  for t,timept in enumerate(timeptList):
    tpID = "%s-%s" % (subjID, timept)
    runCmd += '-tp %s ' % tpID
  runCmd += ' -all'

  qsubCmd = 'qsub -S /bin/bash -l tmem=%.1fG -l h_vmem=%.1fG -l h_rt=%d:%d:0 -N %s -j y -wd %s ' % (MEM_LIMIT, MEM_LIMIT, WALL_TIME_LIMIT_HOURS, WALL_TIME_LIMIT_MIN, jobName, OUTPUT_DIR)

  cmdClust = 'echo "%s" | %s' % (runCmd, qsubCmd) # echo the matlab cmd then pipe it to qsub
  return cmdClust, runCmd

def getQsubCmd3(timept, t1_file, subjID):
  timeptID = "%s-%s" % (subjID, timept)
  templateID = "template_%s" % subjID
  jobName = 'A3_%s' % timeptID
  runCmd = '%s; %s;    %s/bin/recon-all -long %s %s -all' % (
  exportFreeSurfCmd, exportSubjCmd, freesurfPath, timeptID, templateID)

  qsubCmd = 'qsub -S /bin/bash -l tmem=%.1fG -l h_vmem=%.1fG -l h_rt=%d:%d:0 -N %s -j y -wd %s ' % (
  MEM_LIMIT, MEM_LIMIT, WALL_TIME_LIMIT_HOURS, WALL_TIME_LIMIT_MIN, jobName, OUTPUT_DIR)

  cmdClust = 'echo "%s" | %s' % (runCmd, qsubCmd)  # echo the matlab cmd then pipe it to qsub
  return cmdClust, runCmd, timeptID, templateID, jobName

def getQsubCmd4(timept, t1_file, subjID):
  timeptID = "%s-%s" % (subjID, timept)
  templateID = "template_%s" % subjID
  jobName = 'A4_%s' % timeptID
  runCmd = '%s; %s;    %s/bin/recon-all -long %s %s -qcache -no-isrunning' % (
  exportFreeSurfCmd, exportSubjCmd, freesurfPath, timeptID, templateID)

  qsubCmd = 'qsub -S /bin/bash -l tmem=%.1fG -l h_vmem=%.1fG -l h_rt=%d:%d:0 -N %s -j y -wd %s ' % (
  MEM_LIMIT, MEM_LIMIT, WALL_TIME_LIMIT_HOURS, WALL_TIME_LIMIT_MIN, jobName, OUTPUT_DIR)

  cmdClust = 'echo "%s" | %s' % (runCmd, qsubCmd)  # echo the matlab cmd then pipe it to qsub
  return cmdClust, runCmd, timeptID, templateID, jobName

def getQsubCmdPET1(timept, subjID):
  timeptID = "%s-%s" % (subjID, timept)
  templateID = "template_%s" % subjID
  subjID = '%s.long.%s' % (timeptID, templateID)
  jobName = 'P1_%s' % timeptID
  runCmd = '%s; %s;  %s/bin/gtmseg --s %s' % (exportFreeSurfCmd, exportSubjCmd,
                                                freesurfPath, subjID)

  qsubCmd = 'qsub -S /bin/bash -l tmem=%.1fG -l h_vmem=%.1fG -l h_rt=%d:%d:0 -N %s -j y -wd %s ' % (
  MEM_LIMIT, MEM_LIMIT, WALL_TIME_LIMIT_HOURS, WALL_TIME_LIMIT_MIN, jobName, OUTPUT_DIR)

  cmdClust = 'echo "%s" | %s' % (runCmd, qsubCmd)  # echo the matlab cmd then pipe it to qsub
  return cmdClust, runCmd, timeptID, templateID, jobName

sub_list = [x for x in os.listdir(rawMriPath) if os.path.isdir(os.path.join(rawMriPath, x))]
print(len(sub_list))

TEST_RUN = args.test
if TEST_RUN:
  partIndices = [1]
else:
  partIndices = range(len(sub_list))

for p in partIndices:
  subject = sub_list[p]
  subDir = os.listdir(os.path.join(rawMriPath, subject))[0]
  print('subDir', subDir)
#for subject in [sub_list[1]]:
  sub_path = os.path.join(rawMriPath, subject + '/%s' % subDir)
  #print(sub_path)
  timepts = os.listdir(sub_path)
  files = glob.glob("*.nii")
  timepts.sort() # make sure timepoints are in the right order

  if TEST_RUN:
    tpIndices = [0]
  else:
    tpIndices = range(len(timepts))

  print('processing part %d/%d' % (p, len(sub_list)))

  for t in tpIndices:
    tp = timepts[t]
    #for tp in [timepts[0]]:
    #print(tp)
    fld = os.listdir(sub_path + '/' + tp)
    t1_file = glob.glob('%s/%s/%s/*.nii' % (sub_path, tp, fld[0]))[0]
    #print(t1_file)

    (clustCmd1, simpleCmd1) = getQsubCmd1(tp, t1_file, subject)
    # print(simpleCmd1)
    print(clustCmd1)
    if step == 1 and not args.printOnly:
      os.system(clustCmd1)

    ################# part 3 #####################

    (clustCmd3, simpleCmd3, timeptID, templateID, jobName) = getQsubCmd3(tp, t1_file, subject)
    # print(simpleCmd3)
    #print(clustCmd3)

    # os.system(simpleCmd3)

    # logFiles = glob.glob('%s/part3/%s*' % (OUTPUT_DIR, jobName))
    #
    # if len(logFiles) > 1:
    #   print("warning: 2 log files", logFiles)
    #
    # output = qx('tail -n 1 %s' % logFiles[0], shell=True)
    #
    # if output.decode("utf-8")  != 'done\n':
    #   print('running job %s' % jobName)
    #

    if step == 3 and not args.printOnly:
      #print('rm -rf %s/%s.long.%s' % (SUBJECTS_DIR, timeptID, templateID))
      #os.system('rm -rf %s/%s.long.%s' % (SUBJECTS_DIR, timeptID, templateID))
      os.system(clustCmd3)

    ################# part 4 #####################

    (clustCmd4, simpleCmd4, _, _, jobName4) = getQsubCmd4(tp, t1_file, subject)

    # logFiles = glob.glob('%s/part4/%s*' % (OUTPUT_DIR, jobName4))
    # #print('%s/part4/%s*' % (OUTPUT_DIR, jobName4), logFiles)
    #
    # if len(logFiles) > 1:
    #   print("warning: 2 log files", logFiles)
    #
    # output = qx('tail -n 1 %s' % logFiles[0], shell=True)
    #
    # if output.decode("utf-8") != 'done\n':
    #   #print('running job %s' % jobName4)
    #
    print(clustCmd4)
    if step == 4:
      # os.system(simpleCmd4)
      os.system(clustCmd4)

    thFiles = glob.glob('%s/%s.long.%s/surf/*h.thickness.fwhm0.fsaverage.mgh' %
                        (SUBJECTS_DIR, timeptID, templateID))

    #print('%s/%s.long.%s/surf/*h.thickness.fwhm0.fsaverage.mgh' %
    #                    (SUBJECTS_DIR, timeptID, templateID))
    #print(thFiles)
    #print(ads)
    # assert(len(thFiles) <= 2)
    # if len(thFiles) < 2:
    #   print("warning: th files missing for %s.long.%s" % (timeptID, templateID))
    #   # from part 337 onwards
    #
    #   print(clustCmd4)
    #   if step == 4 and subject != '023_S_0926':
    #     #print(simpleCmd4)
    #     #os.system(simpleCmd4)
    #     os.system(clustCmd4)

    clustCmdP1, simpleCmdP1, _, _, _ = getQsubCmdPET1(tp, subject)
    # print(clustCmdP1)
    #print(simpleCmdP1)
    if step == 5 and not args.printOnly:
      os.system(clustCmdP1)

  (clustCmd2, simpleCmd2) = getQsubCmd2(timepts, subject)
  # print(simpleCmd2)
  #print(clustCmd2)
  if step == 2 and not args.printOnly:
    os.system(clustCmd2)

