import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve

import sys


from pathlib import Path

# Get the current working directory as a Path object
current_dir = Path.cwd()
parent_dir = current_dir.parent
input_folders_files = os.path.join(parent_dir, 'full_column_layer23')
sys.path.append(input_folders_files)


# INPUTS
DURATION_SIMULATIONS = 3000  # in ms
DT_DESIRED = 0.05  # in ms -> this is the resolution of the simualtion in V1 model in bmtk
DT_DOWNSAMPLED = 10  # in ms
V_REST = -72.5 # target membrane potential assumed when the kernels were estimated
NUM_CHANNELS = 26 # number of channels
TARGET_VECTOR = ['e4other'] # target family that generates the LFP in L4. It can easily applied also to the toher neuronal families in L4 which contributes little though
NEURON_TYPE_ID = os.listdir('./kernels/gratings/e4other/') # the folders where the kernels are stored
NUM_LEVELS = 2



spikes_tables_V1 = pd.read_pickle('./Allen_Inst_V1_model_features/L4_spiking_gratings/spike_fullcolumn_L4_gratings.pkl')
spikes_tables_LGN = pd.read_pickle(os.path.join(parent_dir, 'full_column_layer23', 'Allen_Inst_V1_model_features', 'L23_gratings', 'spike_lgn_gratings.pkl'))
spikes_tables_BKG = pd.read_pickle(os.path.join(parent_dir, 'full_column_layer23', 'Allen_Inst_V1_model_features', 'L23_gratings', 'spike_bkg_gratings.pkl'))
spikes_tables_FB = pd.read_pickle(os.path.join(parent_dir, 'full_column_layer23', 'Allen_Inst_V1_model_features', 'L23_gratings', 'spike_fb_gratings.pkl'))


def load_h5_data(file_path):
    """
    Load data from an HDF5 file and compute median membrane potentials.
    """
    with h5py.File(file_path, 'r') as h5_file:
        data = h5_file['report']['v1']['data'][:]
        data_node_ids = h5_file['report']['v1']['mapping']['node_ids'][:]
    return np.nanmedian(data, axis=1), data_node_ids


def downsample_and_repeat(data, dt_original, dt_target):
    """
    Downsample and repeat the data for matching time resolution.
    """
    factor = int(dt_target / dt_original)
    downsampled = data[::factor]
    return np.repeat(downsampled, factor)


def get_spike_rate(times, dt=2**-7, duration=DURATION_SIMULATIONS):
    """
    Compute spike rate as a histogram over given time bins.
    """
    bins = (np.arange(0, duration / dt + 1) * dt - dt / 2)
    hist, _ = np.histogram(times, bins=bins)
    return bins, hist.astype(float)


def interpolate_kernel(kernel, dt_target, dt_original=2**-7, duration=300):
    """
    Interpolate the kernel for finer time resolution.
    """
    time_points = np.arange(0, duration + dt_original, dt_original)
    target_points = np.arange(0, duration + dt_original, dt_target)
    interpolated = np.zeros((kernel.shape[0], len(target_points)))
    for i in range(kernel.shape[0]):
        interpolated[i, :] = interp1d(time_points, kernel[i, :], fill_value="extrapolate")(target_points)
    return interpolated


def compute_lfp(spike_rate, kernels, mem_potentials, t_axis, e_syn, vector_time_normalized_levels, v_rest=V_REST):
    """
    Compute LFP contributions from spike rates and kernels.
    """
    lfp = np.zeros((NUM_CHANNELS, len(t_axis)))
    for ch in range(NUM_CHANNELS):
        for time_idx, t_val in enumerate(t_axis):
            kernel_temp = kernels[int(vector_time_normalized_levels[time_idx])][ch, :]
    
            if(time_idx>0): 
                if(mem_potentials[time_idx]!=mem_potentials[time_idx-1]): 
                    kernel_temp = (kernel_temp/(v_rest-e_syn))*(mem_potentials[time_idx]-e_syn)
                    risultato_temp = fftconvolve(spike_rate, kernel_temp, 'same')
            else: 
                kernel_temp = (kernel_temp/(v_rest-e_syn))*(mem_potentials[time_idx]-e_syn)
                risultato_temp = fftconvolve(spike_rate, kernel_temp, 'same')

            lfp[ch, time_idx] = risultato_temp[time_idx]
    return lfp

        


def process_target_folder(folder, target, t_axis):
    """
    Process a folder for a specific target to compute LFP.
    """
    folder_path = os.path.join(folder, target)
    subfolders = os.listdir(folder_path)
    lfp_dict = {}
    
    for outer_idx, subfolder in enumerate(subfolders, 1):
        
        # Load membrane potentials
        h5_path = './inputs to estimate LFP from kernels/gratings/mem_e4other_'+str(subfolder)+'.h5'
        mem_potentials, _ = load_h5_data(h5_path)
        mem_potentials = downsample_and_repeat(mem_potentials, DT_DESIRED, DT_DOWNSAMPLED) # this vector describes the average membrane potential of e23 cells in the single-layer L23 model
        
        
        # Load vector levels
        vector_levels_path = './kernels/gratings/'+str(target)+'/'+str(subfolder)+'/vector_time_normalized_levels'
        with open(vector_levels_path, 'rb') as file:
            vector_time_normalized_levels = pickle.load(file) #this vector descibes which discrete level of kernels I have to use in each simulated time poitn
    
        
        
        print(f"Processing target {target} folder {outer_idx}/{len(subfolders)}")
        subfolder_path = os.path.join(folder_path, subfolder, '0')
        files = os.listdir(subfolder_path)
        
        lfp_dict_target = {}
        
        for file_idx, file_name in enumerate(files, 1):
            
            print(f"Processing file {file_idx}/{len(files)} in subfolder {outer_idx}")
            # Load kernel data
            kernel_dict = dict()
            for lvl in range(NUM_LEVELS): 
                kernel_path = os.path.join(folder_path, subfolder, str(lvl), file_name)
                with open(kernel_path, 'rb') as file:
                    kernel_data = pickle.load(file)['kernel']['GaussCylinderPotential']
                interpolated_kernels = interpolate_kernel(kernel_data, DT_DESIRED)
                kernel_dict[lvl] = interpolated_kernels
                
            
            if(sum(spikes_tables_V1['pop_name'] == file_name)):# parliamo di un nodo in v1
                spike_rows = spikes_tables_V1[spikes_tables_V1['pop_name'] == file_name]
                lunghezze_spikes = [len(v['spike_times']) for i,v in spike_rows.iterrows() ]
                if(sum(lunghezze_spikes)): 
                    if(len(spike_rows) == 1):
                        spikes = spike_rows['spike_times'].item()
                    else: 
                        spikes = []
                        for ii, vv in spike_rows.iterrows(): 
                            spikes = spikes+vv['spike_times']
                            
                    _, s_rate = get_spike_rate(spikes, DT_DESIRED)
            elif(sum(spikes_tables_LGN['pop_name'] == file_name)): 
                spike_rows = spikes_tables_LGN[spikes_tables_LGN['pop_name'] == file_name]
                lunghezze_spikes = len(spike_rows)
                if(lunghezze_spikes):
                    spikes = list(spike_rows['spike_times'])
                
                    _, s_rate = get_spike_rate(spikes, DT_DESIRED)

            elif(file_name == 'BKG'): 
                spike_rows = spikes_tables_BKG
                lunghezze_spikes = len(spike_rows['timestamps'])
                if(lunghezze_spikes): 
                    spikes = list(spike_rows['timestamps'])
        
                    _, s_rate = get_spike_rate(spikes, DT_DESIRED)
                    
            elif(file_name == 'FB'): 
                spike_rows = spikes_tables_FB
                lunghezze_spikes = len(spike_rows['timestamps'])
        
                if(lunghezze_spikes): 
                    spikes = list(spike_rows['timestamps'])
                    
                    _, s_rate = get_spike_rate(spikes, DT_DESIRED)
                    
            
            if(file_name.startswith('i') or file_name.startswith('LIFi')): 
                e_syn = -70
            else: 
                e_syn = 0
                
            # Compute LFP
            lfp = compute_lfp(s_rate, kernel_dict, mem_potentials, t_axis, e_syn, vector_time_normalized_levels)  # Example E_syn=0
            lfp_dict_target[file_name] = lfp
            
        lfp_dict[subfolder] = lfp_dict_target
        
    return lfp_dict


def main():
    """
    Main function to process data and compute LFPs.
    """
    # Process targets
    for target in TARGET_VECTOR:    

        # Time axis
        t_axis = np.arange(0, DURATION_SIMULATIONS, DT_DESIRED)

        folder = './kernels/gratings/' #the folder in which the kernels are stored

        target_lfp_dict = process_target_folder(folder, target, t_axis)

        # Save results
        output_path = f"./from kernels to LFPs/gratings/dictionary_LFP_{target}.p"
        with open(output_path, 'wb') as output_file:
            pickle.dump(target_lfp_dict, output_file)


if __name__ == "__main__":
    main()



#%% second part of the code -> let's compute the LFPs

import os
import pickle
import numpy as np

# File path for the dictionary containing LFP data
dizionario_files = 'dictionary_LFP_e4other.p'
dizionario_path = f"./from kernels to LFPs/gratings/{dizionario_files}"

# Extract the target name from the filename
target = dizionario_files.split('_')[2].split('.')[0]

# Load the dictionary containing LFP data
with open(dizionario_path, "rb") as file:
    dizionario_LFP_target = pickle.load(file)

# List of keys (subfolders) in the dictionary
chiavi = list(dizionario_LFP_target.keys())

# Process each folder (key) in the dictionary
for outer_loop, folder_key in enumerate(chiavi, start=1):
    # Initialize total LFP array for the current folder
    sample_key = next(iter(dizionario_LFP_target[folder_key]))  # Any key to get dimensions
    sample_LFP = dizionario_LFP_target[folder_key][sample_key]
    LFP_total = np.zeros(sample_LFP.shape)  # Shape based on one of the entries

    # Get the LFP dictionary for the current folder
    dizionario_LFP = dizionario_LFP_target[folder_key]

    # Sum LFP contributions from all sources
    for channel_idx in range(LFP_total.shape[0]):
        print(f"{target} {outer_loop} of {len(chiavi)} processing channel {channel_idx + 1} of {LFP_total.shape[0]}")
        for source, lfp_data in dizionario_LFP.items():
            LFP_total[channel_idx, :] += lfp_data[channel_idx, :]

    # Save the total LFP to a new pickle file
    output_file = f"./from kernels to LFPs/gratings/LFPs_{folder_key}_e4other.p"
    with open(output_file, "wb") as output:
        pickle.dump(LFP_total, output)

#%% Third part of the code -> let's now plot the comparison between the kernel-
# based LFP with the one computed with MC simulation in the Allen V1 model

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.signal as ss

# Initialize lists to store maximum values
massimi_allen = []
massimi_kernel = []

# Define constants
dt = 0.05  # ms
t_vector_full = np.arange(0, 3000, dt)  # Full time vector

NEURON_TYPE_ID = os.listdir('./kernels/gratings/e4other/') # the folders where the kernels are stored


electrode_positions = np.linspace(0., -1000., 26) + 205  # Electrode depths
dz = np.abs(electrode_positions[1] - electrode_positions[0])  # Spacing between electrodes
scalebars = 0.01  # Scale bar size (1 μV)
fs = 1000 / dt  # Sampling frequency (Hz)
fc = 300  # Cut-off frequency for low-pass filter
fc_digital = fc / (fs / 2)  # Normalized frequency for filtering
b, a = ss.butter(4, fc_digital, 'low')  # Low-pass filter coefficients

# Loop over simulation IDs
for ind1 in NEURON_TYPE_ID: #the target ID of the e4other family in the single-layer L23 model
    # Load LFP data
    lfp_path = f"./from kernels to LFPs/gratings/LFPs_{ind1}_e4other.p"
    LFP_kernels = pickle.load(open(lfp_path, "rb"))
    
    
    with open('./from kernels to LFPs/gratings/LFP_allen_dict.pkl', 'rb') as file:
        LFP_allen_dict = pickle.load(file)    
    LFP_allen = LFP_allen_dict[ind1]
    
        
    # Apply low-pass filter
    LFP_allen = ss.filtfilt(b, a, LFP_allen)
    LFP_kernels = ss.filtfilt(b, a, LFP_kernels)

    # Let's compute the trial average: each trial lasts 1 second
    res_allen = [LFP_allen[:, int(i * (1.0 / (dt * 10**-3))):int((i + 1) * (1.0 / (dt * 10**-3)))] for i in range(3)]
    res_kernels = [LFP_kernels[:, int(i * (1.0 / (dt * 10**-3))):int((i + 1) * (1.0 / (dt * 10**-3)))] for i in range(3)]
    LFP_allen = sum(res_allen) / len(res_allen)
    LFP_kernels = sum(res_kernels) / len(res_kernels)

    # Update time vector for 1-second window
    t_vector = np.arange(0, 1000, dt)

    # Compute maxima for normalization
    massimi_allen.append(np.max(np.abs(LFP_allen)))
    massimi_kernel.append(np.max(np.abs(LFP_kernels)))

    # Normalize signals
    norm_bmtk = np.max(np.abs(LFP_allen))
    norm_k = 2.8 * np.max(np.abs(LFP_kernels))  # Adjust factor if needed

    # Create figure and subplots
    fig = plt.figure(figsize=[10, 10])
    ax1 = fig.add_subplot(141, xlim=[0, 1000])
    ax2 = fig.add_subplot(142, sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(143, sharex=ax1, sharey=ax1)
    ax4 = fig.add_subplot(144)

    # Initialize R-squared list
    R_squared = []

    # Plotting and analysis for each electrode
    for elec_idx in range(len(electrode_positions)):
        # Normalize and center signals
        tmp_allen = LFP_allen[elec_idx] / norm_bmtk
        tmp_allen -= np.mean(tmp_allen)
        
        tmp_kernels = LFP_kernels[elec_idx] / norm_k
        tmp_kernels -= np.mean(tmp_kernels)

        # Plot signals
        ax1.plot(t_vector, tmp_allen * dz + electrode_positions[elec_idx], 'k')
        ax1.set_xlabel('Time [ms]')
        ax1.set_ylabel('Electrode depth (μm)')
        ax1.set_title('Allen Simulations')

        ax2.plot(t_vector, tmp_kernels * dz + electrode_positions[elec_idx], 'r')
        ax2.set_xlabel('Time [ms]')
        ax2.set_title('Kernel Estimates')

        ax3.plot(t_vector, tmp_allen * dz + electrode_positions[elec_idx], 'k')
        ax3.plot(t_vector, tmp_kernels * dz + electrode_positions[elec_idx], 'r')
        ax3.set_xlabel('Time [ms]')
        ax3.set_title('Comparison')

        # Add scalebars
        y_start = 0 * dz + electrode_positions[elec_idx]
        y_end = (scalebars / norm_bmtk) * dz + electrode_positions[elec_idx]
        ax1.plot([100, 100], [y_start, y_end], 'k')

        # Compute R-squared
        cov_mat = np.cov(tmp_allen[t_vector > 100], tmp_kernels[t_vector > 100])
        R_squared.append((cov_mat[0, 1]**2) / (cov_mat[0, 0] * cov_mat[1, 1]))

    # Plot R-squared values
    ax4.plot(R_squared, electrode_positions, '*')
    ax4.set_xlim(-0.1, 1.1)
    ax4.set_title('R-squared')
    ax4.set_yticklabels([])

    plt.tight_layout()
    plt.show()

