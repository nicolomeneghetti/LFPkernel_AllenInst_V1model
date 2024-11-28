import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve

# INPUTS
DURATION_SIMULATIONS = 7500  # in ms
DT_DESIRED = 0.1  # in ms -> this is the resolution of the simualtion in V1 model in bmtk
DT_DOWNSAMPLED = 10  # in ms
V_REST = -72.5 # target membrane potential assumed when the kernels were estimated
NUM_CHANNELS = 26 # number of channels
TARGET_VECTOR = ['e23Cux2'] # target family that generates the LFP in the single layer L23 model

spikes_tables_V1 = pd.read_pickle('./Allen_Inst_V1_model_features/only_L23_spiking_flashes/spike_only_L23_flashes.pkl')
spikes_tables_LGN = pd.read_pickle('./Allen_Inst_V1_model_features/only_L23_spiking_flashes/spike_lgn_flashes.pkl')
spikes_tables_BKG = pd.read_pickle('./Allen_Inst_V1_model_features/only_L23_spiking_flashes/spike_bkg_flashes.pkl')
spikes_tables_FB = pd.read_pickle('./Allen_Inst_V1_model_features/only_L23_spiking_flashes/spike_fb_flashes.pkl')


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

        


def process_target_folder(folder, target, vector_time_normalized_levels, mem_potentials, t_axis):
    """
    Process a folder for a specific target to compute LFP.
    """
    folder_path = os.path.join(folder, target)
    subfolders = os.listdir(folder_path)
    lfp_dict = {}
    
    for outer_idx, subfolder in enumerate(subfolders, 1):
        
        print(f"Processing target {target} folder {outer_idx}/{len(subfolders)}")
        subfolder_path = os.path.join(folder_path, subfolder, '0')
        files = os.listdir(subfolder_path)
        
        for file_idx, file_name in enumerate(files, 1):
            
        
            print(f"Processing file {file_idx}/{len(files)} in subfolder {outer_idx}")
            # Load kernel data
            kernel_path = os.path.join(subfolder_path, file_name)
            with open(kernel_path, 'rb') as file:
                kernel_data = pickle.load(file)['kernel']['GaussCylinderPotential']
            interpolated_kernels = interpolate_kernel(kernel_data, DT_DESIRED)
            
            
            
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
            lfp = compute_lfp(s_rate, interpolated_kernels, mem_potentials, t_axis, e_syn, vector_time_normalized_levels)  # Example E_syn=0
            lfp_dict[file_name] = lfp
    
    return lfp_dict


def main():
    """
    Main function to process data and compute LFPs.
    """
    # Load membrane potentials
    h5_path = './inputs_to_estimate_LFPs/mem_e23Cux2_487661754_flashes.h5'
    mem_potentials, _ = load_h5_data(h5_path)
    mem_potentials = downsample_and_repeat(mem_potentials, DT_DESIRED, DT_DOWNSAMPLED) # this vector describes the average membrane potential of e23 cells in the single-layer L23 model

    # Time axis
    t_axis = np.arange(0, DURATION_SIMULATIONS, DT_DESIRED)

    # Load vector levels
    vector_levels_path = './kernels_flashes/vector_time_normalized_levels'
    
    
    with open(vector_levels_path, 'rb') as file:
        vector_time_normalized_levels = pickle.load(file) #this vector descibes which discrete level of kernels I have to use in each simulated time poitn

    # Process targets
    for target in TARGET_VECTOR:
        folder = './kernels_flashes/' #the folder in which the kernels are stored
        target_lfp_dict = process_target_folder(folder, target, vector_time_normalized_levels, mem_potentials, t_axis)

        # Save results
        output_path = f"./from kernels to LFPs/flashes/dictionary_LFP_{target}.p"
        with open(output_path, 'wb') as output_file:
            pickle.dump(target_lfp_dict, output_file)


if __name__ == "__main__":
    main()



#%% second part of the code -> let's compute the LFPs

import os
import pickle
import numpy as np

# File path for the dictionary containing LFP data
dizionario_files = 'dictionary_LFP_e23Cux2.p'
dizionario_path = f"./from kernels to LFPs/flashes/{dizionario_files}"

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
    output_file = f"./from kernels to LFPs/flashes/LFPs_{folder_key}_e23Cux2.p"
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
dt = 0.1  # ms
t_vector_full = np.arange(0, 7500, dt)  # Full time vector
electrode_positions = np.linspace(0., -1000., 26) + 205  # Electrode depths
dz = np.abs(electrode_positions[1] - electrode_positions[0])  # Spacing between electrodes
scalebars = 0.01  # Scale bar size (1 μV)
fs = 1000 / dt  # Sampling frequency (Hz)
fc = 300  # Cut-off frequency for low-pass filter
fc_digital = fc / (fs / 2)  # Normalized frequency for filtering
b, a = ss.butter(4, fc_digital, 'low')  # Low-pass filter coefficients

# Loop over simulation IDs
for ind1 in ['487661754']: #the target ID of the e23Cux2 family in the single-layer L23 model
    # Load LFP data
    lfp_path = f"./from kernels to LFPs/flashes/LFPs_{ind1}_e23Cux2.p"
    LFP_kernels = pickle.load(open(lfp_path, "rb"))
    
    h5_file_path = './from kernels to LFPs/flashes/Allen_Inst_V1_ecp_contributions_node_type_id.h5' # this is the LFPs computed by the Allen Institute
    with h5py.File(h5_file_path, 'r') as h5_file:
        LFP_allen = h5_file[ind1][:].T

    # Apply low-pass filter
    LFP_allen = ss.filtfilt(b, a, LFP_allen)
    LFP_kernels = ss.filtfilt(b, a, LFP_kernels)

    # Let's compute the trial average: each trial lasts 750 ms
    res_allen = [LFP_allen[:, int(i * (0.75 / (dt * 10**-3))):int((i + 1) * (0.75 / (dt * 10**-3)))] for i in range(10)]
    res_kernels = [LFP_kernels[:, int(i * (0.75 / (dt * 10**-3))):int((i + 1) * (0.75 / (dt * 10**-3)))] for i in range(10)]
    LFP_allen = sum(res_allen) / len(res_allen)
    LFP_kernels = sum(res_kernels) / len(res_kernels)

    # Update time vector for 1-second window
    t_vector = np.arange(0, 750, dt)

    # Compute maxima for normalization
    massimi_allen.append(np.max(np.abs(LFP_allen)))
    massimi_kernel.append(np.max(np.abs(LFP_kernels)))

    # Normalize signals
    norm_bmtk = np.max(np.abs(LFP_allen))
    norm_k = 2.8 * np.max(np.abs(LFP_kernels))  # Adjust factor if needed

    # Create figure and subplots
    fig = plt.figure(figsize=[10, 10])
    ax1 = fig.add_subplot(141, xlim=[0, 750])
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

