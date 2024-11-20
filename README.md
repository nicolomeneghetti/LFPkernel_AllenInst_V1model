# LFPkernel_AllenInst_V1model
This folder contains the code for estimating the kernels used for LFP estimation in a large-scale V1 mouse model.

The neuronal dynamics were simulated from a mouse V1 column model. The V1 model codes and files are freely available and can be found here: https://doi.org/10.5061/dryad.k3j9kd5b8. 
Details can be found in the original paper: Rimehaug, Atle E., et al. "Uncovering circuit mechanisms of current sinks and sources with biophysical simulations of primary visual cortex." elife 12 (2023): e87169. doi: https://doi.org/10.7554/eLife.87169. The LFPs from this model were simulated with (i) multicompartmental simulations; and (ii) with a computationally efficient kernel-based approach. 

The kernel based methods to estimate LFPs from firing rates was firt presented in "Hagen, Espen, et al. "Brain signal predictions from multi-scale networks using a linearized framework." PLOS Computational Biology 18.8 (2022): e1010353. doi: https://doi.org/10.1371/journal.pcbi.1010353" and available here: https://github.com/LFPy/LFPykernels

