
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

parser = argparse.ArgumentParser(description='The script merges all the cortical thickess images '
 'into one matrix and also assembles the extra information. '
 'It uses the ADNIMERGE spreadsheet from ADNI.')

parser.add_argument('--printOnly', action="store_true", help='only print experiment to be run, not actualy run it')

parser.add_argument('--test', action="store_true", help='only for testing one subject')

parser.add_argument('--fwhmLevel', dest="fwhmLevel", type=int, default=0,
                    help='full-width half max level: 0, 5, 10, 15, 20 or 25')

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


SUBJECTS_DIR = '%s/ADNI_data/MP-Rage_proc_all/subjects' % homeDir
rawMriPath = '%s/ADNI_data/MP-Rage_proc_all/ADNI' % homeDir
ADNI2_DIR =  '%s/ADNI_data/ADNI2_MAYO/subjects' % homeDir

# MEM_LIMIT = 7.9 # in GB
REPO_DIR = '%s/phd_proj/voxelwiseDPM' % homeDir
OUTPUT_DIR = '%s/clusterOutputADNI' % REPO_DIR

def getAgeFromBl(ageAtBl, visitCode):
  if visitCode == 'bl':
    return ageAtBl
  elif visitCode[0] == 'm':
    return ageAtBl + float(visitCode[1:])/12
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

def getCogTest(cogStr):
  try:
    return float(cogStr)
  except ValueError:
    return np.nan


def parseDX(dxChange, dxCurr, dxConv, dxConvType, dxRev):
  # returns (ADNI1_diag, ADNI2_diag) as a pair of integers

  dxChangeToCurrMap = [0, 1,2,3,2,3,3,1,2,1] # 0 not used
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

def parseDiagMerge(diagStr):
  # print('diagStr', diagStr) adDxCha[]  a
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

def filterArrays(subjID, rid, visit, acqDate, gender, age, scanTimepts, diagADNI1, diagADNI2, mask):
  # filter those with less than 4 visits
  subjID = subjID[mask]
  rid = rid[mask]
  visit = visit[mask]
  acqDate = [acqDate[i] for i in range(mask.shape[0]) if mask[i]]
  gender = gender[mask]
  age = age[mask]
  scanTimepts = scanTimepts[mask]
  diagADNI1 = diagADNI1[mask]
  diagADNI2 = diagADNI2[mask]

  assert len(acqDate) == rid.shape[0]
  assert diagADNI1.shape[0] == rid.shape[0]


  return subjID, rid, visit, acqDate, gender, age, scanTimepts, diagADNI1, diagADNI2

with open('../data/ADNI/ADNIMERGE.csv', 'r') as f:
  reader = csv.reader(f)
  rows = [row for row in reader]
  rows = rows[1:] # ignore first line which is the column titles
  nrRows = len(rows)
  # important to include itemsize, otherwise each string will have the size of one byte
  ptidMerge = np.chararray(nrRows, itemsize=20, unicode=False)
  ridMerge = np.zeros(nrRows, int)
  ageMerge = np.zeros(nrRows, np.float16)
  visitCodeMerge = np.chararray(nrRows, itemsize=20, unicode=False)
  apoeMerge = np.zeros(nrRows, int)
  cogTestsMergeLabels = ['cdrsob', 'adas13', 'mmse', 'ravlt']
  cogTestsMerge = np.nan * np.ones((nrRows,4), float)
  examDateMerge = [0 for x in range(nrRows)]
  diagMerge = np.zeros(nrRows, int)
  for r in range(nrRows):
    ridMerge[r] = int(rows[r][0])
    examDateMerge[r] = datetime.strptime(rows[r][6], '%Y-%m-%d')
    #ptidMerge[r] = rows[r][1]
    #visitCodeMerge[r] = rows[r][2]
    #ageMerge[r] = getAgeFromBl(float(rows[r][8]), visitCodeMerge[r])
    #print(r, rows[r][14])
    apoeMerge[r] = getApoe(rows[r][14])
    cogTestsMerge[r,0] = getCogTest(rows[r][18])
    cogTestsMerge[r,1] = getCogTest(rows[r][20])
    cogTestsMerge[r,2] = getCogTest(rows[r][21])
    cogTestsMerge[r,3] = getCogTest(rows[r][22])
    diagMerge[r] = parseDiagMerge(rows[r][7])

    #print(rid[:10],ptid[:10], visitCode[:10], age[:10],gender[:10])
  #print(asdsa)

with open('../data/ADNI/mpr_proc_info_11_08_2016.csv', 'r') as f:
  reader = csv.reader(f)
  rows = [row for row in reader]
  rows = rows[1:] # ignore first line which is the column titles
  nrRows = len(rows)
  subjID = np.chararray(nrRows, itemsize=20, unicode=False)
  rid = np.zeros(nrRows, int)
  acqDate = [0 for x in range(nrRows)]
  gender = np.zeros(nrRows, float)
  ageApprox = np.zeros(nrRows, np.float16) # actually age at first scan, usually baseline but not necessarily
  visit = np.zeros(nrRows, int)
  for r in range(nrRows):
    subjID[r] = rows[r][1]
    rid[r] = int(rows[r][1].split('_')[-1])
    visit[r] = int(rows[r][5])
    acqDate[r] = datetime.strptime(rows[r][9], '%m/%d/%Y')
    gender[r] = getGenderID(rows[r][3], 'M', 'F')
    ageApprox[r] = np.float16(rows[r][4]) # age at bl only


# Find imageIDs of those visits which have not been downloaded
with open('../data/ADNI/MRI_PART2_2_02_2017.csv', 'r') as f:
  reader = csv.reader(f)
  rows = [row for row in reader]
  rows = rows[1:] # ignore first line which is the column titles
  nrRows = len(rows)
  subjIDp2 = np.chararray(nrRows, itemsize=20, unicode=False)
  visitp2 = np.zeros(nrRows, float)
  imgIDp2 = np.zeros(nrRows, int)
  ridp2 = np.zeros(nrRows, int)
  genderp2 = np.zeros(nrRows, float)
  ageApproxp2 = np.zeros(nrRows, np.float16)
  acqDatep2 = [0 for x in range(nrRows)]
  for r in range(nrRows):
    imgIDp2[r] = int(rows[r][0])
    subjIDp2[r] = rows[r][1]
    visitp2[r] = int(rows[r][5])
    acqDatep2[r] = datetime.strptime(rows[r][9], '%m/%d/%Y')
    ridp2[r] = int(rows[r][1].split('_')[-1])
    genderp2[r] = getGenderID(rows[r][3], 'M', 'F')
    ageApproxp2[r] = np.float16(rows[r][4])  # age at bl only

rid = np.concatenate((rid, ridp2), axis=0)
subjID = np.concatenate((subjID, subjIDp2), axis=0)
visit = np.concatenate((visit, visitp2), axis=0)
acqDate = acqDate + acqDatep2
gender = np.concatenate((gender, genderp2), axis=0)
ageApprox =np.concatenate((ageApprox, ageApproxp2), axis=0)

# calculate unrounded ageAtScan and also scan timepoints
scanTimepts = np.zeros(rid.shape[0], int)
age = np.zeros(rid.shape[0], np.float16)
unqRid = np.unique(rid)
nrUnqPart = unqRid.shape[0]

# some patients have multiple scans for the same timepoint, only use the latest scan
# get scan timepoints using acquisition date, not visit (which contains duplicates)
dupVisitsMask = np.zeros(rid.shape[0], bool)
for r, ridCurr in enumerate(unqRid):
  acqDateCurrAll = [acqDate[i] for i in np.where(rid == ridCurr)[0]]
  visitsCurrPart = visit[rid == ridCurr]
  ridsCurrAll = rid[rid == ridCurr]
  ageCurrAll = ageApprox[rid == ridCurr]
  sortedInd = np.argsort(acqDateCurrAll)
  timeSinceBl = [(date - acqDateCurrAll[sortedInd[0]]).days/365 for date in acqDateCurrAll]
  age[rid == ridCurr] = ageCurrAll[sortedInd[0]] + timeSinceBl
  invSortInd = np.argsort(sortedInd)
  scanTimepts[rid == ridCurr] = (invSortInd + 1) # maps from sorted space back to long space

  visitsSorted = visitsCurrPart[sortedInd]
  dupVisitsInd = np.zeros(len(visitsSorted), bool)
  for i in range(0,len(visitsSorted)-1):
    if visitsSorted[i+1] == visitsSorted[i]:
      dupVisitsInd[i] = 1

  #print(dupVisitsInd)
  dupVisitsMask[rid == ridCurr] = dupVisitsInd[invSortInd]

#remove duplicated visits from cross data
diagADNI2 = np.zeros(rid.shape[0], int)  # ADNI 2 style
diagADNI1 = np.zeros(rid.shape[0], int)  # ADNI 1 style
notDupVisitMask = np.logical_not(dupVisitsMask)
(subjID, rid, visit, acqDate, gender, age, scanTimepts, diagADNI1, diagADNI2) = filterArrays(
  subjID, rid, visit, acqDate, gender, age, scanTimepts, diagADNI1, diagADNI2, notDupVisitMask)
ageApprox = ageApprox[notDupVisitMask]

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

twoMoreMask = np.in1d(rid, twoMoreScanRIDs)
print('nr of twoMoreMask', np.sum(twoMoreMask))
threeMoreMask = np.in1d(rid, threeMoreScanRIDs)
print('nr of threeMoreMask', np.sum(threeMoreMask))
fourMoreMask = np.in1d(rid, fourMoreScanRIDs)
print('nr of fourMoreMask', np.sum(fourMoreMask))
fiveMoreMask = np.in1d(rid, fiveMoreScanRIDs)
print('nr of fiveMoreMask', np.sum(fiveMoreMask))
print(rid)
# print(asdsa)

# eliminate those with less than 4 visits and with no matching diagnosis
# attribNotFoundMask = diagADNI1 != 0
# print(attribNotFoundMask.shape, fourMoreMask.shape, diagADNI1.shape)
# filterMask = np.logical_and(attribNotFoundMask, fourMoreMask)
nrMinScans = 3
if nrMinScans == 3:
  filterMask = threeMoreMask
else:
  filterMask = fourMoreMask
# print('rid.shape before ', rid.shape)
(subjID, rid, visit, acqDate, gender, age, scanTimepts, diagADNI1, diagADNI2) = filterArrays(subjID, rid, visit,
   acqDate, gender, age, scanTimepts, diagADNI1, diagADNI2, filterMask)
ageApprox = ageApprox[filterMask]

assert(all(rid != 0))
assert(all(visit != 0))
assert(all([acqDate[i] != 0 for i in range(len(acqDate))]))
assert(not any(np.isnan(gender)))

# add the data from ADNIMERGE - apoe and cog tests
nrSubjCross = rid.shape[0]
cogTests = np.nan * np.ones((nrSubjCross,4), np.float16)
apoe = np.nan * np.ones(nrSubjCross, np.float16)
diag = np.nan * np.ones(nrSubjCross, np.float16)
for s in range(nrSubjCross):
  currSubjAcqDate = acqDate[s]

  maskMerge = ridMerge == rid[s]
  currSubjExamDatesMerge = [examDateMerge[i] for i in range(maskMerge.shape[0]) if maskMerge[i]]
  matchIndex = np.argmin(np.abs([(date - currSubjAcqDate).days for date in currSubjExamDatesMerge]))

  currSubjCogTests = cogTestsMerge[maskMerge]
  currSubjAPOE = apoeMerge[maskMerge]
  currSubjDiag = diagMerge[maskMerge]

  cogTests[s] = currSubjCogTests[matchIndex,:]
  apoeMerge[s] = currSubjAPOE[matchIndex]
  diag[s] = currSubjDiag[matchIndex]

np.random.seed(2)
idx = np.random.permutation(range(rid.shape[0]))[:15]
# print(rid)
# print(idx)
print('rid[idx]', rid[idx])
print('diag[idx]', diag[idx])
print('visit[idx]', visit[idx])
print('age[idx]', age[idx])
print('ageAtBl[idx]', ageApprox[idx])
print(list(zip(age[idx], ageApprox[idx])))
# print(adsads)
ageNotMatchMask = np.abs(age - ageApprox) >= 1.3
print('ageNotMatchMask', ageNotMatchMask)
print('rid[ageNotMatchMask]', rid[ageNotMatchMask])
print('visit[ageNotMatchMask]', visit[ageNotMatchMask])
print('age[ageNotMatchMask]', age[ageNotMatchMask])
print('ageApprox[ageNotMatchMask]', ageApprox[ageNotMatchMask])


ridTest = 985
print('ridTest', ridTest)
print('rid[rid == ridTest]', rid[rid == ridTest])
print('visit[rid == ridTest]', visit[rid == ridTest])
print('age[rid == ridTest]', age[rid == ridTest])
print('ageApprox[rid == ridTest]', ageApprox[rid == ridTest])
print(np.sum((np.abs(age - ageApprox) >= 1.3)))
# assert (np.abs(age - ageApprox) < 2).all()

# print('rid.shape after ', rid.shape)
# print(cogTests[4:10,:], rid[4:10], acqDate[4:10])
print('nrSubjCross', nrSubjCross)
# print(sdas)

# create long data once again for going through the scans
unqRid = np.unique(rid)
scanTimeptLong = []
subjIDLong = []
acqDateLong = []
ridLong = []
for r, ridCurr in enumerate(unqRid):
  currInd = rid == ridCurr
  #print(currInd)
  #print(ads)
  scanTimeptLong += [scanTimepts[currInd]]
  subjIDLong += [subjID[currInd][0]]
  acqDateLong += [[acqDate[i] for i in np.where(currInd)[0]]]
  ridLong += [rid[currInd][0]]

sub_list = [x for x in os.listdir(rawMriPath) if os.path.isdir(os.path.join(rawMriPath, x))]
print(len(sub_list), rid.shape[0])

#check one th file to find dimensions
oneThFile ='%s/002_S_0295-2006-04-18_08_20_30.0.long.template_002_S_0295/surf/lh.thickness.fwhm0.fsaverage.mgh' % SUBJECTS_DIR
oneThObj = nib.freesurfer.mghformat.load(oneThFile)
nrVertices = oneThObj.dataobj.shape[0]
print(oneThObj.dataobj.shape, nrVertices)

nrSubjCross = rid.shape[0]
# avghData = np.nan * np.ones((nrSubjCross, nrVertices), dtype=np.float16) # left hemishpere
avghData = [None for x in range(nrSubjCross)]
# print(asdas)

#bhData = np.nan * np.ones((nrSubjCross, 2*nrVertices), float) # both hemishperes
#scanTimepts = np.nan * np.ones((nrSubjCross,1))
#partCode = np.nan * np.ones((nrSubjCross,1))
#ageAtScan = np.nan * np.ones((nrSubjCross,1))
#diag = np.nan * np.ones((nrSubjCross,1))

subjDirs = os.listdir(SUBJECTS_DIR)
longSubjDirs = [d for d in subjDirs if len(d.split('.')) == 4 ]
fullSubjDirs = ['%s/%s' % (SUBJECTS_DIR, d) for d in longSubjDirs]

subjDirs2 = os.listdir(ADNI2_DIR)
longSubjDirs2 = [d for d in subjDirs2 if len(d.split('.')) == 4 ]
fullSubjDirs += ['%s/%s' % (ADNI2_DIR, d) for d in longSubjDirs2]

# print(adsa)
runPart = 'L'
tmpDataFile = '../data/ADNI/tmpADNIMerge.npz'
if runPart == 'R':
  for p in range(len(subjIDLong)):

    scanTimeptsCurr = scanTimeptLong[p]

    print('processing part %d/%d   %s' % (p, len(subjIDLong), subjIDLong[p]))

    for t in range(len(scanTimeptsCurr)):

      nameStart = '%s-%s' % (subjIDLong[p].decode("utf-8"), acqDateLong[p][t].strftime('%Y-%m-%d'))
      # print('processing ', nameStart)
      # print('fullSubjDirs[:3]', fullSubjDirs[:3])

      matchDirs = []
      for d in fullSubjDirs:
        if d.split('/')[-1].startswith(nameStart):
          matchDirs += [d]

      #print('longSubjDirs', longSubjDirs)
      #print(matchDirs)
      if subjIDLong[p].decode("utf-8") == '023_S_0926':
        print('no dir found for %s' % nameStart)
        continue

      if nameStart == '023_S_1126-2006-12-05':
        continue

      if len(matchDirs) == 0:
        print('no dir found for %s' % nameStart)
        # assert False
        continue

      # if len(matchDirs) > 0:
      #   assert False

      if len(matchDirs) > 1:
        #print('more than one dir matches:', matchDirs)
        matchDirs.sort()

      finalDir = matchDirs[-1]

      indexMask = np.logical_and(rid == ridLong[p], scanTimepts == scanTimeptsCurr[t])
      indexCrossArray = np.where(indexMask)[0]

      if indexCrossArray.shape[0] == 0:
        print('no match found with cross-sectional data for rid %d scanTimepts' % ridLong[p],
          scanTimeptsCurr[t])
        continue

      assert(indexCrossArray.shape[0] == 1)
      indexCross = indexCrossArray[0]
      #print('indexCross', indexCross, 'scanTimepts[indexCross]', scanTimepts[indexCross], 'tp', scanTimeptsCurrPart[p], scanTimeptsCurrPart[tp])
      assert (scanTimepts[indexCross] == scanTimeptsCurr[t])
      assert (rid[indexCross] == ridLong[p])

      lhThFile = '%s/surf/lh.thickness.fwhm%d.fsaverage.mgh'\
                 % (matchDirs[0], args.fwhmLevel)
      rhThFile = '%s/surf/rh.thickness.fwhm%d.fsaverage.mgh' \
                 % (matchDirs[0], args.fwhmLevel)

      print(lhThFile)

      if os.path.isfile(lhThFile) and os.path.isfile(rhThFile):
        lhThObj = nib.freesurfer.mghformat.load(lhThFile)
        #print(lhThObj.dataobj.shape)
        avghData[indexCross] = np.squeeze(lhThObj.dataobj)

        rhThObj = nib.freesurfer.mghformat.load(rhThFile)
        avghData[indexCross] += np.squeeze(rhThObj.dataobj)

        #bhData[indexCross, :] = np.concatenate((lhData[indexCross,:], rhData[indexCross,:]), axis=0)
      else:
        print('file %s not found' % lhThFile)

  tmpDataStruct = dict(avghData=avghData)
  pickle.dump(tmpDataStruct, open(tmpDataFile, 'wb'),
    protocol=pickle.HIGHEST_PROTOCOL)
else:
  tmpDataStruct = pickle.load(open(tmpDataFile, 'rb'))
  avghData = tmpDataStruct['avghData']

assert cogTests.shape[0] == rid.shape[0]

print('avghData', avghData)
notNanInd = np.array([avghData[i] is not None for i in range(len(avghData))])
# also remove all the subjects with only one timepoint
# this needs to be done again as many images are not matched
# (some images with missing tags)
filterInd = notNanInd
ridNN = rid[notNanInd]
unqRidNN = np.unique(ridNN)
for r, ridCurr in enumerate(unqRidNN):
  currInd = rid == ridCurr

  # remove subject if it has only one nonNan visit
  if np.sum(np.logical_and(currInd, notNanInd)) < nrMinScans:
    filterInd[currInd] = 0

notNanInd = filterInd
nrNotNanSubj = np.sum(notNanInd)
print('np.where(np.logical_not(notNanInd))', np.where(np.logical_not(notNanInd)))
print('notNanInd.all()', notNanInd.all())
print('notNanInd', notNanInd)
print(nrNotNanSubj)

avghDataArray = np.zeros((nrNotNanSubj, nrVertices), dtype=np.float16)
counter = 0
for s in range(len(avghData)):
  if notNanInd[s]:
    avghDataArray[counter,:] = avghData[s]
    counter += 1

avghDataArray.astype(np.float16)

print(len(acqDate), avghDataArray.shape[0], notNanInd.shape[0])
assert len(acqDate) == notNanInd.shape[0]

acqDateNotNan = [acqDate[i] for i in range(len(notNanInd)) if notNanInd[i]]
print(len(acqDateNotNan), avghDataArray.shape[0])
assert len(acqDateNotNan) == len(avghDataArray)

avghDataArray /=  2 # divide by 2 to get mean, as so far we have the sum here.

#notNanInd = np.logical_not(np.in1d(partCode, partCode[rowIndMisUnq]))
pointIndices = np.array(range(avghDataArray.shape[1]))
dataStruct = dict(avghData=avghDataArray, pointIndices=pointIndices)#, rhData=rhData[notNanInd,:])

print('avghData dtype', avghDataArray.dtype)
print('age dtype', age.dtype)
print('cogTests dtype', cogTests.dtype)

infoStruct = dict(partCode=rid[notNanInd], studyID=subjID[notNanInd], scanTimepts=scanTimepts[notNanInd],
                  ageAtScan=age[notNanInd], diag=diag[notNanInd], gender=gender[notNanInd],
                  acqDate=acqDateNotNan, visit=visit[notNanInd],
                  cogTests=cogTests[notNanInd,:], cogTestsLabels=cogTestsMergeLabels,
                  apoe=apoe[notNanInd])

print('avghData.shape', avghDataArray.shape)

cortThickADNIData = '../data/ADNI/cortThickADNI%dScansData.npz' % nrMinScans
pickle.dump(dataStruct, open(cortThickADNIData, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

cortThickADNIInfo = '../data/ADNI/cortThickADNI%dScansInfo.npz' % nrMinScans
pickle.dump(infoStruct, open(cortThickADNIInfo, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
