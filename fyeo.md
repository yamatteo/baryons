Issues
======

#### _For your eyes only_ ####

 - There is a piece ready to apply transformations to the data within the dataset. 

   You can find their _vestigia_ in [at row 66 of this file](https://github.com/yamatteo/baryons/blob/5d158dc0ecaa479304daac4619e3e3a375fe6091/gan/vox2vox.py).

   Do we want/need them?

 - There is now a **monolithic** preprocessor. I think it's better, you can have some insight about its inner workings in the [notebook](https://github.com/yamatteo/baryons/blob/5d158dc0ecaa479304daac4619e3e3a375fe6091/notebook.ipynb).

   Speaking of the dataset... I thought that what came out of illustris were positions of particles relative to the center of mass of the halo. But something does not seem right. Am I mistaken?
   
 - I guess what 'valid' and 'test' datasets are for, but we are not using them; and I don't now how to use them. Any advice on this matter is welcome.

 - I'm trying to understand pytorch usage of memory, because maybe there is something wrong or stupid in our code and lower batch size just avoid the problem.

   At this point it looks like the biggest part of memory is occupied by a `.forward(...)` call and that the same memory is released after a `.backward()` call. Maybe all that space is for gradients?

   But the scale is puzzling: before `.forward(...)` with all the models and the inputs memory is about 230MB, after the `.forward(...)` it skyrockets to 2.1GB.
