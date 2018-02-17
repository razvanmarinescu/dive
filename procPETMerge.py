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
from env import *

parser = argparse.ArgumentParser(description='Launches freesurfer processes for ADNI data on cluster or local machine')

parser.add_argument('--test', action="store_true", help='only for testing one subject')

# parser.add_argument('--fwhmLevel', dest="fwhmLevel", type=int,
#   help='full-width half max level: 0, 5, 10, 15, 20 or 25')

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
  raise ValueError('wrong hostname or cluster flag')


PET_DIR = '%s/ADNI_data/av45_all/ADNI' % homeDir
# folder containing around 1500 MRIs that I downloaded in Jan 2017 when I realised
# the MRI images I had did not mathch the AV45
sub_list_pet = [x for x in os.listdir(PET_DIR) if
  os.path.isdir(os.path.join(PET_DIR, x))]


# adniSubFd = 'MP-Rage_proc_all'
adniSubFd = 'ADNI2_MAYO'
SUBJECTS_DIR = '%s/ADNI_data/%s/subjects' % (homeDir, adniSubFd)
rawMriPath = '%s/ADNI_data/%s/ADNI' % (homeDir, adniSubFd)

MEM_LIMIT = 7.9  # in GB
REPO_DIR = '%s/phd_proj/voxelwiseDPM' % homeDir
OUTPUT_DIR = '%s/clusterOutputADNI' % REPO_DIR


def getAgeFromBl(ageAtBl, visitCode):
  if visitCode == 'bl':
    return ageAtBl
  elif visitCode[0] == 'm':
    return ageAtBl + float(visitCode[1:]) / 12
  else:
    return np.nan


def getGenderID(genderStr, maleStr, femaleStr):
  if genderStr == maleStr:
    return 0
  elif genderStr == femaleStr:
    return 1
  else:
    return np.nan


def getApoe(apoeStr):
  try:
    return int(apoeStr)
  except ValueError:
    return -1

def getDiag(diagStr):
  # print('diagStr', diagStr)
  if diagStr == 'CN':
    return CTL
  elif diagStr == 'EMCI':
    return EMCI
  elif diagStr == 'LMCI':
    return LMCI
  elif diagStr == 'AD':
    return AD
  elif diagStr == 'SMC':
    return SMC
  else:
    raise ValueError('diag string has to be CN, EMCI, LMCI or AD')


def getCogTest(cogStr):
  try:
    return float(cogStr)
  except ValueError:
    return np.nan


def parseDX(dxChange, dxCurr, dxConv, dxConvType, dxRev):
  # returns (ADNI1_diag, ADNI2_diag) as a pair of integers

  dxChangeToCurrMap = [0, 1, 2, 3, 2, 3, 3, 1, 2, 1]  # 0 not used
  if dxChange:
    # dxChange tring not empty
    adni1_diag = dxChangeToCurrMap[int(dxChange)]
    adni2_diag = dxChange
  else:
    adni1_diag = dxCurr
    if dxConv == '0':
      adni2_diag = int(dxCurr)
    elif dxConv == '1' and dxConvType == '1':
      adni2_diag = 4
    elif dxConv == '1' and dxConvType == '3':
      adni2_diag = 5
    elif dxConv == '1' and dxConvType == '2':
      adni2_diag = 6
    elif dxConv == '2':
      adni2_diag = int(dxRev) + 6
    else:
      return ValueError('wrong values for diagnosis')

  return adni1_diag, adni2_diag


def filterArrays(subjID, rid, visit, acqDate, gender, age, scanTimepts, mask):
  # filter those with less than 4 visits
  subjID = subjID[mask]
  rid = rid[mask]
  visit = visit[mask]
  acqDate = [acqDate[i] for i in range(mask.shape[0]) if mask[i]]
  gender = gender[mask]
  age = age[mask]
  scanTimepts = scanTimepts[mask]
  # diag = diag[mask]

  assert len(acqDate) == rid.shape[0]
  # assert diag.shape[0] == rid.shape[0]

  return subjID, rid, visit, acqDate, gender, age, scanTimepts


with open('../data/ADNI/ADNIMERGE.csv', 'r') as f:
  reader = csv.reader(f)
  rows = [row for row in reader]
  rows = rows[1:]  # ignore first line which is the column titles
  nrRows = len(rows)
  # important to include itemsize, otherwise each string will have the size of one byte
  ptidMerge = np.chararray(nrRows, itemsize=20, unicode=False)
  ridMerge = np.zeros(nrRows, float)
  ageMerge = np.zeros(nrRows, float)
  visitCodeMerge = np.chararray(nrRows, itemsize=20, unicode=False)
  genderMerge = np.zeros(nrRows, int)
  apoeMerge = np.zeros(nrRows, int)
  diagMerge = np.zeros(nrRows, int)
  cogTestsMergeLabels = ['cdrsob', 'adas13', 'mmse', 'ravlt']
  cogTestsMerge = np.nan * np.ones((nrRows, 4), float)
  examDateMerge = [0 for x in range(nrRows)]
  for r in range(nrRows):
    ridMerge[r] = int(rows[r][0])
    examDateMerge[r] = datetime.strptime(rows[r][6], '%Y-%m-%d')
    # ptidMerge[r] = rows[r][1]
    # visitCodeMerge[r] = rows[r][2]
    # ageMerge[r] = getAgeFromBl(float(rows[r][8]), visitCodeMerge[r])
    # genderMerge[r] = getGenderID(rows[r][9], 'Male', 'Female')
    # print(r, rows[r][14])
    apoeMerge[r] = getApoe(rows[r][14])
    diagMerge[r] = getDiag(rows[r][7])
    cogTestsMerge[r, 0] = getCogTest(rows[r][18])
    cogTestsMerge[r, 1] = getCogTest(rows[r][20])
    cogTestsMerge[r, 2] = getCogTest(rows[r][21])
    cogTestsMerge[r, 3] = getCogTest(rows[r][22])

    # print(rid[:10],ptid[:10], visitCode[:10], age[:10],gender[:10])
    # print(asdsa)

with open('../data/ADNI/AV45_processed_all_2_02_2017.csv', 'r') as f:
  reader = csv.reader(f)
  rows = [row for row in reader]
  rows = rows[1:]  # ignore first line which is the column titles
  nrRows = len(rows)
  subjID = np.chararray(nrRows, itemsize=20, unicode=False)
  rid = np.zeros(nrRows, int)
  acqDateMri = [0 for x in range(nrRows)]
  gender = np.zeros(nrRows, float)
  ageAtScanRounded = np.zeros(nrRows, float)  # actually age at first scan, usually baseline but not necessarily
  visit = np.zeros(nrRows, float)
  for r in range(nrRows):
    subjID[r] = rows[r][1]
    rid[r] = int(rows[r][1].split('_')[-1])
    visit[r] = int(rows[r][5])
    acqDateMri[r] = datetime.strptime(rows[r][9], '%m/%d/%Y')
    gender[r] = getGenderID(rows[r][3], 'M', 'F')
    ageAtScanRounded[r] = float(rows[r][4])  # age at bl only

# calculate unrounded ageAtScan and also scan timepoints
scanTimepts = np.zeros(nrRows, int)
age = np.zeros(nrRows, float)
unqRid = np.unique(rid)
nrUnqPart = unqRid.shape[0]

# some patients have multiple scans for the same timepoint, only use the latest scan
# get scan timepoints using acquisition date, not visit (which contains duplicates)
dupVisitsMask = np.zeros(nrRows, bool)
for r, ridCurr in enumerate(unqRid):
  acqDateCurrAll = [acqDateMri[i] for i in np.where(rid == ridCurr)[0]]
  visitsCurrPart = visit[rid == ridCurr]
  ridsCurrAll = rid[rid == ridCurr]
  ageCurrAll = ageAtScanRounded[rid == ridCurr]

  # need to estimate age with decimal precision using ageAtBl and acquisition date
  sortedInd = np.argsort(acqDateCurrAll)
  timeSinceBl = [(date - acqDateCurrAll[sortedInd[0]]).days / 365 for date in acqDateCurrAll]
  age[rid == ridCurr] = ageAtScanRounded[sortedInd[0]] + np.array(timeSinceBl)
  invSortInd = np.argsort(sortedInd)
  scanTimepts[rid == ridCurr] = (invSortInd + 1)  # maps from sorted space back to long space

  visitsSorted = visitsCurrPart[sortedInd]
  dupVisitsInd = np.zeros(len(visitsSorted), bool)
  for i in range(0, len(visitsSorted) - 1):
    if visitsSorted[i + 1] == visitsSorted[i]:
      dupVisitsInd[i] = 1

  # print(dupVisitsInd)
  dupVisitsMask[rid == ridCurr] = dupVisitsInd[invSortInd]

print('dupVisitsMask', dupVisitsMask)
print('np.sum(dupVisitsMask)', np.sum(1- dupVisitsMask))
# remove duplicated visits from cross data
notDupVisitMask = np.logical_not(dupVisitsMask)
(subjID, rid, visit, acqDateMri, gender, age, scanTimepts)\
  = filterArrays(subjID, rid, visit, acqDateMri, gender, age, scanTimepts,
  notDupVisitMask)

# print(subjID[rid == 2], acqDate[rid == 2], visit[rid == 2])
# print(ads)

# filter those with less than 4 visits
unqRid = np.unique(rid)
twoMoreScanRIDs = []  # two or more
threeMoreScanRIDs = []  # three or more
fourMoreScanRIDs = []  # four or more
fiveMoreScanRIDs = []  # five or more
for r, ridCurr in enumerate(unqRid):
  currInd = rid == ridCurr

  nrTimepts = np.sum(rid == ridCurr)
  if nrTimepts >= 2:
    twoMoreScanRIDs += [ridCurr]
  if nrTimepts >= 3:
    threeMoreScanRIDs += [ridCurr]
  if nrTimepts >= 4:
    fourMoreScanRIDs += [ridCurr]
  if nrTimepts >= 5:
    fiveMoreScanRIDs += [ridCurr]

print('total nr of subjects', rid.shape[0])
twoMoreMask = np.in1d(rid, twoMoreScanRIDs)
print('nr of twoMoreMask', np.sum(twoMoreMask))
threeMoreMask = np.in1d(rid, threeMoreScanRIDs)
print('nr of threeMoreMask', np.sum(threeMoreMask))
fourMoreMask = np.in1d(rid, fourMoreScanRIDs)
print('nr of fourMoreMask', np.sum(fourMoreMask))
fiveMoreMask = np.in1d(rid, fiveMoreScanRIDs)
print('nr of fiveMoreMask', np.sum(fiveMoreMask))
# print(asdsa)

# eliminate those with less than 4 visits and with no matching diagnosis
# attribNotFoundMask = diagADNI1 != 0
# print(attribNotFoundMask.shape, fourMoreMask.shape, diagADNI1.shape)
#filterMask = np.logical_and(attribNotFoundMask, fourMoreMask)
filterMask = twoMoreMask
# print('rid.shape before ', rid.shape)
(subjID, rid, visit, acqDateMri, gender, age, scanTimepts) = filterArrays(
  subjID, rid, visit, acqDateMri, gender, age, scanTimepts, filterMask)

assert (all(rid != 0))
assert (all(visit != 0))
assert (all([acqDateMri[i] != 0 for i in range(len(acqDateMri))]))
assert (not any(np.isnan(gender)))

# align the data from ADNIMERGE to the MRI S/S dataset - apoe, cog tests and diag
nrSubjCross = rid.shape[0]
cogTests = np.nan * np.ones((nrSubjCross, 4), float)
apoe = np.nan * np.ones(nrSubjCross, float)
diag = np.nan * np.ones(nrSubjCross, float)

for s in range(nrSubjCross):
  currSubjAcqDate = acqDateMri[s]

  # match entries in ADNIMERGE and MRI S/S by image acquisition date
  maskMerge = ridMerge == rid[s]
  currSubjExamDatesMerge = [examDateMerge[i] for i in range(maskMerge.shape[0]) if maskMerge[i]]
  matchIndex = np.argmin(np.abs([(date - currSubjAcqDate).days for date in currSubjExamDatesMerge]))

  currSubjCogTests = cogTestsMerge[maskMerge]
  currSubjAPOE = apoeMerge[maskMerge]
  currSubjDiag = diagMerge[maskMerge]

  cogTests[s] = currSubjCogTests[matchIndex, :]
  apoe[s] = currSubjAPOE[matchIndex]
  diag[s] = currSubjDiag[matchIndex]

# print('rid.shape after ', rid.shape)
# print(cogTests[4:10,:], rid[4:10], acqDate[4:10])
# print(sdas)

# create long data once again for going through the scans
unqRid = np.unique(rid)
scanTimeptLong = []
subjIDLong = []
acqDateMriLong = []
ridLong = []
for r, ridCurr in enumerate(unqRid):
  currInd = rid == ridCurr
  # print(currInd)
  # print(ads)
  scanTimeptLong += [scanTimepts[currInd]]
  subjIDLong += [subjID[currInd][0]]
  acqDateMriLong += [[acqDateMri[i] for i in np.where(currInd)[0]]]
  ridLong += [rid[currInd][0]]

sub_list = [x for x in os.listdir(rawMriPath) if os.path.isdir(
  os.path.join(rawMriPath, x))]
print(len(sub_list), rid.shape[0])

# check one surface file to find dimensions
oneSfFile = '%s/022_S_0130/AV45_Coreg,_Avg,_Std_Img_and_Vox_Siz,_Uniform_Resolution' \
            '/2011-04-27_16_02_33.0/I232193/lh.mgx.gm.fsaverage.1mm.nii.gz' % PET_DIR
oneSfObj = nib.load(oneSfFile)
# print(oneSfObj)
nrVertices = oneSfObj.dataobj.shape[0]
print(oneSfObj.dataobj.shape, nrVertices)

nrSubjCross = rid.shape[0]
lhData = np.nan * np.ones((nrSubjCross, nrVertices), dtype=np.float16)  # left hemishpere
rhData = np.nan * np.ones((nrSubjCross, nrVertices), dtype=np.float16)

# bhData = np.nan * np.ones((nrSubjCross, 2*nrVertices), float) # both hemishperes
# scanTimepts = np.nan * np.ones((nrSubjCross,1))
# partCode = np.nan * np.ones((nrSubjCross,1))
# ageAtScan = np.nan * np.ones((nrSubjCross,1))
# diag = np.nan * np.ones((nrSubjCross,1))

# print(asdas)

subjDirs = os.listdir(SUBJECTS_DIR)
longSubjDirs = [d for d in subjDirs if len(d.split('.')) == 4]

nrSubjLong = len(subjIDLong)
subjMatched = np.zeros(nrSubjLong)
timeptsMatched = [0 for x in range(nrSubjLong)]

print(subjIDLong[:10])
# print(adsa)

for p in range(len(subjIDLong)):

  scanTimeptsCurr = scanTimeptLong[p]

  sub_path = os.path.join(PET_DIR, subjIDLong[p].decode("utf-8") +
    '/AV45_Coreg,_Avg,_Std_Img_and_Vox_Siz,_Uniform_Resolution')

  timepts = os.listdir(sub_path)
  timepts.sort()  # make sure timepoints are in the right order

  print('processing part %d/%d   %s' % (p, len(subjIDLong), subjIDLong[p]))

  # print(timepts)
  # print(adsas)

  timeptsMatched[p] = []

  for t in range(len(scanTimeptsCurr)):

    tpPET = timepts[t] # as directory, e.g. 023_S_0058-2010-03-19_13_05_44.0
    #for tp in [timepts[0]]:
    #print(tp)
    fld = os.listdir(sub_path + '/' + tpPET)

    petSubjFold = os.listdir('%s/%s' % \
                    (sub_path, tpPET))[0]

    # print(petSubjFold)


    indexMask = np.logical_and(rid == ridLong[p], scanTimepts == scanTimeptsCurr[t])
    indexCrossArray = np.where(indexMask)[0]

    if indexCrossArray.shape[0] == 0:
      print('no match found with cross-sectional data for rid %d scanTimepts' % ridLong[p], scanTimeptsCurr[t])
      continue


    subjMatched[p] = 1
    timeptsMatched[p] += [1]

    # make sure only one entry matches
    assert (indexCrossArray.shape[0] == 1)
    indexCross = indexCrossArray[0]
    # print('indexCross', indexCross, 'scanTimepts[indexCross]',
    #   scanTimepts[indexCross], 'tp')
    assert (scanTimepts[indexCross] == scanTimeptsCurr[t])
    assert (rid[indexCross] == ridLong[p])

    lhFile = '%s/%s/%s/lh.mgx.gm.fsaverage.1mm.nii.gz' % \
                    (sub_path, tpPET, petSubjFold)
    rhFile = '%s/%s/%s/rh.mgx.gm.fsaverage.1mm.nii.gz' % \
             (sub_path, tpPET, petSubjFold)

    # print(lhFile)
    # print(asada)

    if os.path.isfile(lhFile) and os.path.isfile(rhFile):
      lhThObj = nib.load(lhFile)
      # print(lhThObj.dataobj.shape)
      lhData[indexCross, :] = np.squeeze(lhThObj.dataobj)

      rhThObj = nib.load(rhFile)
      rhData[indexCross, :] = np.squeeze(rhThObj.dataobj)

      # bhData[indexCross, :] = np.concatenate((lhData[indexCross,:], rhData[indexCross,:]), axis=0)
    else:
      print('file %s not found' % lhFile)


print('subjMatched', subjMatched)
print('nrSubjMatched %s out of %s', np.sum(subjMatched), subjMatched.shape[0])
print('nrTimeptsMatched %s out of %s', np.sum([np.sum(x) for x in timeptsMatched]),
      np.sum([len(x) for x in timeptsMatched]))

lhData = (lhData + rhData) / 2  # average both hemispheres, don't save in a diff variable so as not to allocate space
print(lhData)
# (rowIndMis, colIndMis) = np.where(np.isnan(lhData))
# rowIndMisUnq = np.unique(rowIndMis)
# print(rowIndMisUnq, rid[rowIndMisUnq])

assert diag.shape[0] == rid.shape[0]
assert cogTests.shape[0] == rid.shape[0]

# notNanInd = np.array([i for i in range(lhData.shape[0]) if i not in rowIndMisUnq])
notNanInd = np.logical_not(np.isnan(lhData[:,0]))

print(len(acqDateMri), lhData.shape[0], np.sum(notNanInd))
# assert len(acqDateMri) == lhData.shape[0]
# assert len(acqDate) == notNanInd.shape[0]

assert not np.isnan(lhData[notNanInd, :]).any()

# also remove all the subjects with only one timepoint
# this needs to be done again as for PET many images are not matched
# (some images with missing tags)
filterInd = notNanInd
ridNN = rid[notNanInd]
unqRidNN = np.unique(ridNN)
for r, ridCurr in enumerate(unqRidNN):
  currInd = rid == ridCurr

  # remove subject if it has only one nonNan visit
  if np.sum(np.logical_and(currInd, notNanInd)) == 1:
    filterInd[currInd] = 0



pointIndices = np.array(range(lhData.shape[1]))
dataStruct = dict(avghData=lhData[filterInd, :], pointIndices=pointIndices)  # , rhData=rhData[notNanInd,:])

infoStruct = dict(partCode=rid[filterInd], studyID=subjID[filterInd],
  scanTimepts=scanTimepts[filterInd],
  ageAtScan=np.array(age[filterInd], dtype=np.float16), diag=diag[filterInd],
  gender=gender[filterInd],
  visit=visit[filterInd],
  cogTests=cogTests[filterInd, :], cogTestsLabels=cogTestsMergeLabels,
  apoe=apoe[filterInd])

print('lhData.shape', lhData.shape, 'lhData[filterInd,:]', lhData[filterInd, :].shape[0])

adniData = '../data/ADNI/av45FWHM0ADNIData.npz'
pickle.dump(dataStruct, open(adniData, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

adniInfo = '../data/ADNI/av45FWHM0ADNIInfo.npz'
pickle.dump(infoStruct, open(adniInfo, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
