This folder contains the download link to the kernels corresponding to the full-column L4 V1 model.

Inside of it you'll find:

The file "vector_time_normalized_levels" contains an array indicating, for each time point, which kernels should be used. In our scenario we used two types of kernels, corresponding to the two discrete levels in which the leak conductance change throughout time was discretized.

The folder 'e4other'... contains the actual kernels. You'll notice it has two subfolders, named '0' and '1': this subfolders indicate the two types of kernels mentioned above.

Unline the e23Cux2 family, the e4other contains a lot of sub-families. You'll notice consequently that the kernels are estimated for each one of these e4other sub-families. 
