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

parser.add_argument('--freesurfVS', dest="freesurfVS", type=int, default=6,
                    help='Freesurfer version, as an int ... default is 5 (i.e. 5.3.0)')

parser.add_argument('--hemi', dest="hemi",
                    help='lh or rh')


args = parser.parse_args()

hostName = gethostname()
print(hostName)

if hostName == 'razvan-Inspiron-5547':
  homeDir = '/home/razvan'
  freesurfPath = '/usr/local/freesurfer-5.3.0'
elif hostName == 'razvan-Precision-T1700':
  homeDir = '/home/razvan'
  freesurfPath = '/usr/local/freesurfer-6.0.0'
elif args.cluster:
  homeDir = '/home/rmarines'
  freesurfPath = '/home/rmarines/src/freesurfer-6.0.0'
else:
  raise ValueError('add freesurfer paths and home directory above, like in the examples.')


MRI_SUBJECTS_DIR_NOV = '%s/ADNI_data/MP-Rage_proc_all/subjects' % homeDir
rawMriPath = '%s/ADNI_data/MP-Rage_proc_all/ADNI' % homeDir
PET_DIR = '%s/ADNI_data/av45_all/ADNI/' % homeDir
# folder containing around 1500 MRIs that I downloaded in Jan 2017 when I realised
# the MRI images I had did not mathch the AV45
MRI_SUBJECTS_DIR_JAN = '%s/ADNI_data/ADNI2_MAYO/subjects' % homeDir
sub_list_pet = [x for x in os.listdir(PET_DIR) if os.path.isdir(os.path.join(PET_DIR, x))]

MEM_LIMIT = 7.9 # in GB
REPO_DIR = '%s/phd_proj/voxelwiseDPM' % homeDir
OUTPUT_DIR = '%s/clusterOutputPET' % REPO_DIR


exportFreeSurfCmd = 'export FREESURFER_HOME=%s; source %s/SetUpFreeSurfer.sh' % (freesurfPath,freesurfPath)

WALL_TIME_LIMIT_HOURS = int(args.timeLimit)
WALL_TIME_LIMIT_MIN = 15
step = int(args.step)

maxYearDeltaPetMri = 1  # maximum time lapse allowed between the PET and MRI scans

def getQsubCmdPET2(timeptMRI, subjID, timeptPET, petImgTemplate, subjDir):
  """
  Register the PET with the corresponding MRI scan

  :param timeptMRI:
  :param subjID:
  :param timeptPET:
  :param petImgTemplate:
  :param subjDir:
  :return:
  """
  timeptID = "%s-%s" % (subjID, timeptMRI)
  templateID = "template_%s" % subjID
  subjIDlong = '%s.long.%s' % (timeptID, templateID)
  jobName = 'P2_%s-%s' % (subjID, timeptPET) # set the jobID to be the subjID + PETtimept
  regFileName = 'petRegistration.reg.lta'
  exportSubjCmd = 'export SUBJECTS_DIR=%s' % subjDir
  registrationFilePath = '/'.join(petImgTemplate.split('/')[:-1]) + '/' + regFileName
  checkCmd = 'tkregisterfv --mov %s --reg %s --surfs' % (petImgTemplate, registrationFilePath)
  runCmd = '%s; %s;  %s/bin/mri_coreg --s %s --mov %s --reg %s' % \
           (exportFreeSurfCmd, exportSubjCmd, freesurfPath, subjIDlong,
            petImgTemplate, registrationFilePath)

  qsubCmd = 'qsub -S /bin/bash -l tmem=%.1fG -l h_vmem=%.1fG -l h_rt=%d:%d:0 -N %s -j y -wd %s ' % (
  MEM_LIMIT, MEM_LIMIT, WALL_TIME_LIMIT_HOURS, WALL_TIME_LIMIT_MIN, jobName, OUTPUT_DIR)

  cmdClust = 'echo "%s" | %s' % (runCmd, qsubCmd)  # echo the matlab cmd then pipe it to qsub
  return cmdClust, runCmd, timeptID, templateID, jobName

def getQsubCmdPET3(timeptMRI, subjID, timeptPET, petImgTemplate, subjDir, currMriTimeptFold):
  """
  Perform Partial Volume Correction

  :param timeptMRI:
  :param subjID:
  :param timeptPET:
  :param petImgTemplate:
  :param subjDir:
  :param currMriTimeptFold:
  :return:
  """
  timeptID = "%s-%s" % (subjID, timeptMRI)
  templateID = "template_%s" % subjID
  jobName = 'P3_%s-%s' % (subjID, timeptPET) # set the jobID to be the subjID + PETtimept
  regFileName = 'petRegistration.reg.lta'
  exportSubjCmd = 'export SUBJECTS_DIR=%s' % subjDir
  imgFolder = '/'.join(petImgTemplate.split('/')[:-1])
  registrationFilePath = imgFolder + '/' + regFileName
  pvcOutFolder = imgFolder + '/gtmpvc_output'
  gtmSegFile = '%s/mri/gtmseg.mgz' % currMriTimeptFold
  FWHM_level = 8 # taken from ADNI-PET preprocessing website, step 4:
  # Each image set is filtered with a scanner-specific filter function (can be a
  # non-isotropic filter) to produce images of a uniform isotropic resolution of
  # 8 mm FWHM, the approximate resolution of the lowest resolution scanners used in ADNI
  runCmd = '%s; %s;  %s/bin/mri_gtmpvc --i %s --reg %s --psf %d --seg %s ' \
           '--default-seg-merge  --auto-mask PSF .01 --mgx .01 --o %s' % \
           (exportFreeSurfCmd, exportSubjCmd, freesurfPath, petImgTemplate,
           registrationFilePath, FWHM_level, gtmSegFile, pvcOutFolder)

  qsubCmd = 'qsub -S /bin/bash -l tmem=%.1fG -l h_vmem=%.1fG -l h_rt=%d:%d:0 -N %s -j y -wd %s ' % (
  MEM_LIMIT, MEM_LIMIT, WALL_TIME_LIMIT_HOURS, WALL_TIME_LIMIT_MIN, jobName, OUTPUT_DIR)

  cmdClust = 'echo "%s" | %s' % (runCmd, qsubCmd)  # echo the matlab cmd then pipe it to qsub
  return cmdClust, runCmd, timeptID, templateID, jobName

def getQsubCmdPET4(timeptMRI, subjID, timeptPET, petImgTemplate, subjDir, hemi):
  ''' Sample the PET volume into anatomical fsaverage volume via the individual subject's MRI scan

  :param timeptMRI:
  :param subjID:
  :param timeptPET:
  :param petImgTemplate:
  :param subjDir:
  :param hemi
  :return:
  '''

  timeptID = "%s-%s" % (subjID, timeptMRI)
  templateID = "template_%s" % subjID
  jobName = 'P3_%s-%s' % (subjID, timeptPET) # set the jobID to be the subjID + PETtimept
  exportSubjCmd = 'export SUBJECTS_DIR=%s' % subjDir
  imgFolder = '/'.join(petImgTemplate.split('/')[:-1])
  pvcOutFolder = imgFolder + '/gtmpvc_output'
  srcImgFile = '%s/mgx.ctxgm.nii.gz' % pvcOutFolder
  regFile = '%s/aux/bbpet2anat.lta' % pvcOutFolder
  #srcImgFile = petImgTemplate
  # regFile = imgFolder + '/petRegistration.reg.lta'
  outFile = '%s/%s.mgx.gm.fsaverage.1mm.nii.gz' % (imgFolder, hemi)
  runCmd = '%s; %s;  %s/bin/mri_vol2surf --mov %s --reg %s --hemi %s ' \
           '--projfrac-avg 0.2 0.8 0.1 --o %s --cortex --trgsubject fsaverage' % \
           (exportFreeSurfCmd, exportSubjCmd, freesurfPath, srcImgFile,
           regFile, hemi, outFile)

  qsubCmd = 'qsub -S /bin/bash -l tmem=%.1fG -l h_vmem=%.1fG -l h_rt=%d:%d:0 -N %s -j y -wd %s ' % (
  MEM_LIMIT, MEM_LIMIT, WALL_TIME_LIMIT_HOURS, WALL_TIME_LIMIT_MIN, jobName, OUTPUT_DIR)

  cmdClust = 'echo "%s" | %s' % (runCmd, qsubCmd)  # echo the matlab cmd then pipe it to qsub
  return cmdClust, runCmd, timeptID, templateID, jobName


#mri_vol2surf --mov mgx.ctxgm.nii.gz --reg aux/bbpet2anat.lta --hemi lh --projfrac 0.5 --o lh.mgx.ctxgm.fsaverage.sm00.nii.gz --cortex --trgsubject fsaverage

def findClosestMRIFold(SUBJECTS_DIR, subjectIDPET, tpPET, SUBJECTS_DIR2=None):

  subjLongMriFlds = [os.path.join(SUBJECTS_DIR, x) for x in os.listdir(SUBJECTS_DIR)
    if x.startswith(subjectIDPET) and 'long' in x]

  if SUBJECTS_DIR2 is not None:
    subjLongMriFlds += [os.path.join(SUBJECTS_DIR2, x) for x in os.listdir(SUBJECTS_DIR2)
      if x.startswith(subjectIDPET) and 'long' in x]

  if len(subjLongMriFlds) == 0:
    return None, None

  subjLongMriFlds.sort()
  # print('subjLongMriFlds', subjLongMriFlds)
  datesLong = [datetime.strptime(x.split('/')[-1][11:21], '%Y-%m-%d') for x in subjLongMriFlds]
  # print('datesLong', datesLong)

  timeDiffs = [float(abs((d - datetime.strptime(tpPET[:10], '%Y-%m-%d')).days))
               / 365 for d in datesLong]

  # print(timeDiffs)
  minTIndex = np.argmin(timeDiffs)

  if timeDiffs[minTIndex] < maxYearDeltaPetMri:
    currTimeptFold = subjLongMriFlds[minTIndex]
    tpMRI = '.'.join((currTimeptFold.split('/')[-1][11:]).split('.')[:2])
    # print(tpPET)
    # print(tpMRI)
    # print(currTimeptFold)
    # print(asdasd)
  else:
    currTimeptFold = None
    tpMRI = None

  return currTimeptFold, tpMRI

TEST_RUN = args.test
if TEST_RUN:
  partIndices = [0]
else:
  partIndices = range(len(sub_list_pet))

subjMatched = np.zeros(len(partIndices))
timeptsMatched = [0 for x in partIndices]

mriNotProcFile = 'mriNotProc.npz'
dataStruct = pickle.load(open(mriNotProcFile, 'rb'))

unqSubjIdp2 = dataStruct['unqSubjIdp2']
unqVisitp2 = dataStruct['unqVisitp2']
unqImgIDp2 = dataStruct['unqImgIDp2']
unqAcqDatesp2 = dataStruct['unqAcqDatesp2']

finalImgIDs = []

for p in partIndices:
  subjectIDPET = sub_list_pet[p]
#for subject in [sub_list_pet[1]]:
  sub_path = os.path.join(PET_DIR, subjectIDPET + '/AV45_Coreg,_Avg,_Std_Img_and_Vox_Siz,_Uniform_Resolution')
  #print(sub_path)
  timepts = os.listdir(sub_path)
  timepts.sort() # make sure timepoints are in the right order

  if TEST_RUN:
    tpIndices = [0]
  else:
    tpIndices = range(len(timepts))

  print('processing part %d/%d   %s' % (p, len(sub_list_pet), sub_list_pet[p]))

  timeptsMatched[p] = []

  for t in tpIndices:
    tpPET = timepts[t] # as directory, e.g. 023_S_0058-2010-03-19_13_05_44.0
    #for tp in [timepts[0]]:
    #print(tp)
    fld = os.listdir(sub_path + '/' + tpPET)

    petSubjFold = os.listdir('%s/%s' % \
                    (sub_path, tpPET))[0]
    petSubjFiles = '%s/%s/%s/*.dcm' % \
                    (sub_path, tpPET, petSubjFold)

    # print('petSubjFiles', petSubjFiles)
    # currTimeptFold = findClosestPETTimept(petSubjFiles, tp)
    currMriTimeptFold, currTpMRI = findClosestMRIFold(MRI_SUBJECTS_DIR_JAN, subjectIDPET, tpPET, MRI_SUBJECTS_DIR_NOV)
    # currMriTimeptFold = /home/razvan/ADNI_data/ADNI2_MAYO/subjects/072_S_4694-2012-06-15_13_03_43.0.long.template_072_S_4694
    # currTpMRI = 2012-06-15_13_03_43.0


    visitCurrSubjInd = np.where([x.decode('ascii') == subjectIDPET for x in unqSubjIdp2])[0]
    mriDatesCurrSubj = [unqAcqDatesp2[i] for i in visitCurrSubjInd]
    unqImgIDCurrSubj = [unqImgIDp2[i] for i in visitCurrSubjInd]

    # if mriImgIndex is None:# and currTimeptFold is None:
    if currMriTimeptFold is None:
      timeptsMatched[p] += [0]
      print('MRI not matched for subj %s' % subjectIDPET)
      continue

    # finalImgIDs += [unqImgIDCurrSubj[mriImgIndex]]

    subjMatched[p] = 1
    timeptsMatched[p] += [1]

    # print('currTimeptFold', currTimeptFold)
    av45DCMFiles = glob.glob(petSubjFiles)

    createNiiCmd = '%s/bin/dcm2niix %s' % (homeDir, av45DCMFiles[0])
    petFileNiiStr = '%s/%s/%s/*.nii' % (sub_path, tpPET, petSubjFold)
    # petImgPrev = glob.glob(petFileNiiStr)
    # if step == 1 and not args.printOnly and len(petImgPrev) == 0:
      # os.system(createNiiCmd)


    # print(petFileNiiStr)
    petImgTemplate = glob.glob(petFileNiiStr)[0]

    if step == 1:

      if len(petImgTemplate) == 0:
        print('No nii pet img %s', petFileNiiStr)

    # print(petFileNiiStr)
    # print(petImgTemplate)
    currSubjDir = '/'.join(currMriTimeptFold.split('/')[:-1])
    subjectIDMRI = subjectIDPET # same subj ID for both MRI and PET images
    (clustCmdP2, simpleCmdP2, timeptID, templateID, jobName) = \
      getQsubCmdPET2(currTpMRI, subjectIDMRI, tpPET, petImgTemplate, currSubjDir)
    if step == 2:
      print(simpleCmdP2)
      if not args.printOnly:
        #os.system(simpleCmdP2)
        os.system(clustCmdP2)


    (clustCmdP3, simpleCmdP3, _, _, _) = \
      getQsubCmdPET3(currTpMRI, subjectIDMRI, tpPET, petImgTemplate, currSubjDir, currMriTimeptFold)
    if step == 3:
      print(simpleCmdP3)
      if not args.printOnly:
        os.system(clustCmdP3)

    #for hemi in ['lh', 'rh']:
    for hemi in [args.hemi]:
      (clustCmdP4, simpleCmdP4, _, _, _) = \
        getQsubCmdPET4(currTpMRI, subjectIDMRI, tpPET, petImgTemplate, currSubjDir, hemi)
      if step == 4:
        print(simpleCmdP4)
        if not args.printOnly:
          os.system(simpleCmdP4)

# finalImgIDs = list(np.unique(finalImgIDs))

# for p in range(len(finalImgIDs)):
#   print('%d,' % finalImgIDs[p],end='')

print('subjMatched', subjMatched)
print('nrSubjMatched %s out of %s', np.sum(subjMatched), subjMatched.shape[0])
print('nrTimeptsMatched %s out of %s', np.sum([np.sum(x) for x in timeptsMatched]),
      np.sum([len(x) for x in timeptsMatched]))
# print('finalImgIDs', len(finalImgIDs))

