Issues
======

#### _For your eyes only_ ####

 - When we build halos files we save the ids and we put dark matter and baryons in the same csv: why not two separate files? Later we read the ids and load the files one by one. We can even voxelize on the fly in the dataset.

   Even better: why can't we build a single numpy array with dimensions `[halo_id, dm/gas, x, y, z]`? If we can put it whole into memory we can avoid i/o operation, which are of course slower.
 - In dataset we unsqueeze the data to add a fictitious _channels_ dimension, and there ~~is~~ was even an option in the parser about how many channels we have. I get that Conv3D, BatchNorm and such expect data to have a _channels_ dimension, but do we need the options? Are we ever going to have more than one channel?
 - There is a piece ready to apply transformations to the data within the dataset. Do we want them?
 - I suddently realized that also Machine Learning is tainted with colonialism: a piece of our neurotic network is called `discriminator`!
 - We have 'train' , 'valid' and 'test' dataloaders but we use only the first. What about the other two?