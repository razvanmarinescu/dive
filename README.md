== Getting started ==

=== Synthetic tests ===

Run the DIVE model on several synthetic tests using the following command:

make synthAll 

This will run three main synthetic scenarios:
1. cluster trajectories getting closer to one another 
2. increasing number of cluster
3. decreasing number of subjects 

As the scenarios get harder and harder (e.g. very few subjects generated), DIVE will perform worse.

 === Real data ===

In order to get DIVE working on real data, you need to use some images where the voxels are aligned across subjects and timepoints. You can use any software you prefer for the image registration (I used Freesurfer). When applying to a different dataset, DIVE might require setting suitable priors and initialisations of the parameters.

== Implementation details ==

The main class of the model is in voxelDPM.py. The launcher scripts used to start various experiments are launcher[*].py, and the VDPM[*].py are different flavours of the model, which can use linear trajectories, allow missing data or perform spatial regularisation with a Markov Random Field. I recommend you use the VDPMMean (model index 8), because it uses a mathematical optimisation that yields a x20 decrease in computation time. I am currently finishing the journal paper describing all of these.

== References ==

A vertex clustering model for disease progression: Application to cortical thickness images Razvan V. Marinescu, Arman Eshaghi, Marco Lorenzi, Alexandra L. Young, Neil P. Oxtoby, Sara Garbarino, Timothy J. Shakespeare, Sebastian J. Crutch and Daniel C. Alexander http://www.homepages.ucl.ac.uk/~rmaprvm/MarinescuIPMI2017.pdf


