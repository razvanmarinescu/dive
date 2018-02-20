step2:
	python3 procADNICluster.py --cluster --timeLimit 40 --step 2 --printOnly --freesurfVS 6

step4:
	python3 procADNICluster.py --cluster --timeLimit 20 --step 4
lh_m5:
	echo "/share/apps/python-3.5.1/bin/python3 /home/rmarines/phd/mres/voxelwiseDPM/launchADNIthick.py --dataset Lh --firstModel 5 --lastModel 5 --nrOuterIter 30 --nrInnerIter 2 --nrClust 4"| qsub -S /bin/bash -l tmem=31.7G -l h_vmem=31.7G -l h_rt=10:15:0 -N Lh_m5 -j y -wd /home/rmarines/phd_proj/voxelwiseDPM/clusterOutputADNI
blend_col:
	file=resfiles/adniThMo10kCl4_VWDPMLinear/params_o30.npz  blender --background --python blenderCol.py
atrophyMapFVPCA:
	freeview -f /usr/local/freesurfer-5.3.0/subjects/fsaverage/surf/lh.inflated:overlay=resfiles/drc10kavgThickMapFWHM0PCA:overlay_threshold=-1.3,-0.46 --viewport 3d

# canonical correlation between cog tests and vertex-wise cortical thickness measures
cca:
	python3 drcThickCCA.py --dataset avg --fwhmLevel 0  

# run vertex clustering model (standard one)  dataset: ADNI avg - assumes uninformative prior on subject-shift params 
adniThickAvgFWHM0Cl3K-meansRa1Pr0_VWDPMMean:
	python3 launchThick.py --launcherScript adniThick.py --dataset avg --fwhmLevel 0 --firstInstance 0 --lastInstance 0 --nrProc 10 --models 8 --nrOuterIt 15 --nrInnerIt 1 --nrClust 3 --initClustering k-means --rangeFactor 1 --serial

# run vertex clustering model (the ROI average version)  dataset: DRC avg - assumes uninformative prior on subject-shift params 
drcThickAvgFWHM0Cl3HistRa1Pr0_VDPMMean:
	python3 launchThick.py --launcherScript drcThick.py --dataset avg --fwhmLevel 0 --firstInstance 1 --lastInstance 1 --nrProc 1 --models 4 --nrOuterIt 25 --nrInnerIt 1 --nrClust 3 --initClustering hist --rangeFactor 1 --serial

clearJobs:
	qstat | awk '{ if ($$5 == "Eqw") print $$1;}' | xargs qmod -cj

deleteAllJobs:
	qstat | awk '{print $1;}' | xargs qdel


printJobsNotDoneADNI:
	python3 printJobsNotDone.py --clustOutputFolder clusterOutputADNI

printGtmsegNotDoneADNI:
	python3 printJobsNotDone.py --clustOutputFolder clusterOutputADNI --doneStr "gtmseg Done"

colorAdniThickMRF:
	file=resfiles/adniThavgFWHM0Initk-meansCl3Pr0Ra1Mrf5_VDPM_MRF/clust0_adniThavgFWHM0Initk-meansCl3Pr0Ra1Mrf5_VDPM_MRF.npz pngFile=resfiles/adniThavgFWHM0Initk-meansCl3Pr0Ra1Mrf5_VDPM_MRF/clust0_adniThavgFWHM0Initk-meansCl3Pr0Ra1Mrf5_VDPM_MRF.png  blender --background --python colorClustProb.py

adniThickClustList:
	python3 launchThick.py --launcherScript adniThick.py --firstInstance 1 --lastInstance 1 --nrProc 1 --models 9 --nrOuterIt 15 --nrInnerIt 1 --nrClustList 2,3,4,5,6,7,8,9,10,12,15,20,30,40,50,60,70,80,90,100 --initClustering k-means --rangeFactor 0 --informPrior 1 --cluster --timeLimit 23

qstatFullJobName:
	qstat -xml | tr '\n' ' ' | sed 's#<job_list[^>]*>#\n#g'   | sed 's#<[^>]*>##g' | grep " " | column -t

copyThickness:
	rsync -av --include="*h.thickness.fwhm*" --include="*/" --exclude="*" "rmarines@comic100:/home/rmarines/VWDPM/ADNI_data/MP-Rage_proc_all/subjects/" . 

adniPetManyClust:
	python3 launchThick.py --launcherScript adniPet.py --firstInstance 1 --lastInstance 1 --nrProc 1 --models 9 --nrOuterIt 25 --nrInnerIt 1 --nrClustList 18,19,20,21,22,23,24,25,26,27,28,29,30,32,34,36,38,40,42,44,46,48,50 --initClustering k-means --rangeFactor 1 --informPrior 0 --cluster --timeLimit 23

copySlopeCol:
	find . -name "*slopeCol*" | xargs -I {} find {} -name "*.png" | awk '{ split($1,a,"/"); system("cp "$1" "a[2]"/slopeCol.png") }'

# actually not working properly
cmpAnnotAfterRegist:
	cd ~/HCP/Downloads/100206/T1w/100206/label; freeview $SUBJECTS_DIR/100206/mri/T1.mgz 100206_labeledVol.nii.gz -f $SUBJECTS_DIR/100206/surf/lh.white $SUBJECTS_DIR/100206/surf/lh.pial:annot=aligned100206.annot; freeview ../mri/T1.mgz labeledVol.nii.gz -f ../surf/lh.white ../surf/lh.pial:annot=lh.100206.aparc.annot

makeHCPFreeSurfLinks:
	cd /usr/local/freesurfer-6.0.0/subjects; ls -d /media/razvan/Seagate\ Expansion\ Drive/HCP/Downloads/1*// | awk '{split($3,a,"/"); print(a[4])}' | xargs -I {} sudo ln -s /media/razvan/Seagate\ Expansion\ Drive/HCP/Downloads/{}/T1w/{} {}


####### Main experiments for the paper ########
adniMain:
	python3 adniThick.py --fwhmLevel 0 --runIndex 1 --nrProc 1 --models 9 --nrOuterIt 25 --nrInnerIt 1 --nrClust 8 --initClustering  k-means --rangeFactor 1 --informPrior 0;
	
drcAdMain:
	python3 drcThick.py --fwhmLevel 0 --runIndex 1 --nrProc 1 --models 9 --nrOuterIt 25 --nrInnerIt 1 --nrClust 4 --initClustering  k-means --rangeFactor 1 --informPrior 0;
	
drcPcaMain:
	python3 drcThick.py --fwhmLevel 0 --runIndex 1 --nrProc 1 --models 9 --nrOuterIt 25 --nrInnerIt 1 --nrClust 3 --initClustering  k-means --rangeFactor 1 --informPrior 0;

petMain:
	python3 adniPet.py --fwhmLevel 0 --runIndex 1 --nrProc 1 --models 9 --nrOuterIt 25 --nrInnerIt 1 --nrClust 20 --initClustering  k-means --rangeFactor 1 --informPrior 0;

adniMaster:
	python3 adniThick.py --fwhmLevel 0 --runIndex 0 --nrProc 10 --models 9 --nrOuterIt 25 --nrInnerIt 1 --nrClust 8 --initClustering  k-means --rangeFactor 1 --informPrior 0;

petMaster:
	python3 adniPet.py --fwhmLevel 0 --runIndex 0 --nrProc 10 --models 9 --nrOuterIt 25 --nrInnerIt 1 --nrClust 20 --initClustering  k-means --rangeFactor 1 --informPrior 0;


fourMainExp:
	adniMain
	drcAdMain
	drcPcaMain
	petMain


####### 10-fold Cross-Validation experiments on cluster ########

adniThickValidClust:
	echo -n '6,9,10' | xargs -I {} -d, /share/apps/python-3.5.1/bin/python3 launchThick.py --launcherScript adniThick.py --firstInstance 1 --lastInstance 10 --nrProc 10 --models {} --nrOuterIt 25 --nrInnerIt 1 --nrClust 8 --initClustering k-means --rangeFactor 1 --informPrior 0 --cluster --timeLimit 23 --mem 31

adniPetValidClust:
	echo -n '6,9,10' | xargs -I {} -d, /share/apps/python-3.5.1/bin/python3 launchThick.py --launcherScript adniPet.py --firstInstance 1 --lastInstance 10 --nrProc 10 --models {} --nrOuterIt 25 --nrInnerIt 1 --nrClust 20 --initClustering k-means --rangeFactor 1 --informPrior 0 --cluster --timeLimit 23 --mem 31
	

###### HCP connectome analysis ########

createHCPpatches:
	python3 createHCPpatches.py --runPart 10  --maxSearchRadius 5

copyFromCmicPC:
	scp -r 'razvan@128.16.15.167:/home/razvan/phd_proj/voxelwiseDPM/resfiles/*NearNeigh.npz' resfiles;
	echo -n 'adniPetInitk-meansCl20Pr0Ra1Mrf5DataNZ_VDPM_MRF,drcThFWHM0Initk-meansCl4Pr0Ra1Mrf5_VDPM_MRFPCA,drcThFWHM0Initk-meansCl3Pr0Ra1Mrf5_VDPM_MRFAD,adniThFWHM0Initk-meansCl8Pr0Ra1Mrf5_VDPM_MRF' | xargs -I {} -d, scp -r 'razvan@128.16.15.167:/home/razvan/phd_proj/voxelwiseDPM/resfiles/{}/*' {};
	

##### TADPOLE ########

tadpoleLdb:
	python3 tadpole.py --runIndex 1 --nrProc 1 --models 13 --nrOuterIt 5 --nrInnerIt 1 --nrClust 12 --initClustering hist --rangeFactor 1 --informPrior 0 --leaderboard 1

tadpoleD2:
	python3 tadpole.py --runIndex 1 --nrProc 1 --models 13 --nrOuterIt 5 --nrInnerIt 1 --nrClust 12 --initClustering hist --rangeFactor 1 --informPrior 0 --leaderboard 0

tadpoleD3:
	python3 tadpoleD3.py --runIndex 1 --nrProc 1 --models 13 --nrOuterIt 5 --nrInnerIt 1 --nrClust 13 --initClustering hist --rangeFactor 1 --informPrior 0 --leaderboard 0
