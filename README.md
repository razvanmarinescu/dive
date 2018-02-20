Getting started: You can try to get the synthetic results working by running:


make synthAll 


However, in order to get it working on real data, you need to use some images where the voxels are aligned across subjects and timepoints. You can use any software you prefer for the image registration (I used Freesurfer). Also, when applying it to a different dataset, the model might require some tuning of parameters or setting suitable priors.


The main class of the model is in voxelDPM.py. The launcher scripts used to start various experiments are launcher[*].py, and the VDPM[*].py are different flavours of the model, which can use linear trajectories, allow missing data or perform spatial regularisation with a Markov Random Field. I recommend you use the VDPMMean (model index 8), because it uses a mathematical optimisation that yields a x20 decrease in computation time. I am currently finishing the journal paper describing all of these.


Reference paper: http://www.homepages.ucl.ac.uk/~rmaprvm/MarinescuIPMI2017.pdf


