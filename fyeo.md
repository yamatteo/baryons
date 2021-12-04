Issues
======

#### _For your eyes only_ ####

 - When we build halos files we save the ids and we put dark matter and baryons in the same csv: why not two separate files? Later we read the ids and load the files one by one. We can even voxelize on the fly in the dataset.
 - In dataset we unsqueeze the data to add a fictitious _channels_ dimensions, and there is even an option in the parser about how many channels we have. I get that Conv3D, BatchNorm and such expect data to have a channels dimensions, but do we need the options? Are we ever going to have more than one channel?
 - There is a piece ready to apply transformations to the data within the dataset. Do we want them?
 - I suddently realized that also Machine Learning is tainted with colonialism: a piece of our neurotic network is called `discriminator`!
 - We have 'train' , 'valid' and 'test' dataloaders but we use only the first. What about the other two?