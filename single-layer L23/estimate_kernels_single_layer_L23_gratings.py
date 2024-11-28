import os
from os.path import join
import json
import numpy as np
import pandas as pd
import pickle
import scipy.stats as st

from kernel_codes import KernelApprox, GaussCylinderPotential
import utils_functions
from utils_functions import (filter_connectivity, load_dataframe, filter_layer23, compute_population_parameters, 
                             add_mult_params, add_delay_params, get_synapse_type, load_synaptic_params, handle_missing_weight, 
                             process_synaptic_positions, process_synaptic_positions_LGN, get_spike_rate)

#%% this function computes all the information needed to estimate the kernels


def compute_utils_from_bmtk_to_population(target_population: str, target_node_id: int, simulation_length: int, dt_simulation: float, zaxis_rotation: float, num_levels:int):
    # misc parameters
    dt_kernels = 2**-7  # time resolution (ms)
    t_X = 500  # time of synaptic activations (ms)
    tau = 150  # duration of impulse response function after onset (ms)    
    
    
    """Main function to process connectivity and prepare parameters."""
    morphology_dir, dynamics_dir, template_file = utils_functions.initial_setups()

    # Load connectivity data
    connectivity_files = {
        "V1-V1": './Allen_Inst_V1_model_features/features_V1_V1_L23.pkl',
        "LGN-V1": './Allen_Inst_V1_model_features/features_L23V1_LGN.pkl',
        "BKG-V1": './Allen_Inst_V1_model_features/features_L23V1_BKG.pkl',
        "FB-V1": './Allen_Inst_V1_model_features/features_L23V1_FB.pkl',
    }
       
    
    connectivity_L23V1_to_L23V1 = filter_connectivity(load_dataframe(connectivity_files['V1-V1']), target_population, target_node_id)
    connectivity_L23V1_to_L23V1 = filter_layer23(connectivity_L23V1_to_L23V1)
    
    connectivity_LGN_to_L23V1 = filter_connectivity(load_dataframe(connectivity_files['LGN-V1']), target_population, target_node_id)
    connectivity_BKG_to_L23V1 = filter_connectivity(load_dataframe(connectivity_files['BKG-V1']), target_population, target_node_id)
    connectivity_FB_to_L23V1 = filter_connectivity(load_dataframe(connectivity_files['FB-V1']), target_population, target_node_id)



    X = list(connectivity_L23V1_to_L23V1['source_query'])+\
    list(connectivity_LGN_to_L23V1['LGN_pop_name'])+\
    list(connectivity_BKG_to_L23V1['source_query'])+\
    list(connectivity_FB_to_L23V1['source_query']) # contains the list of the presynaptic families

    N_X = np.array(list(connectivity_L23V1_to_L23V1['numerosity_source_nodes'])+\
                   list(connectivity_LGN_to_L23V1['numerosity_source'])+\
                   list(connectivity_BKG_to_L23V1['numerosity_source_nodes'])+\
                   list(connectivity_FB_to_L23V1['numerosity_source_nodes'])) # presynpatic population sizes


    Y = target_population # postsynaptic population
    N_Y = int(list(connectivity_L23V1_to_L23V1['numerosity_target_nodes'])[0]) # postsynaptic population size

    g_eff = True  # account for changes in passive leak due to persistent synaptic activations


    C_YX = list(connectivity_L23V1_to_L23V1['conn_probability'])+\
    list(connectivity_LGN_to_L23V1['conn_probability'])+\
    list(connectivity_BKG_to_L23V1['conn_probability'])+\
    list(connectivity_FB_to_L23V1['conn_probability'])# contains the list of connection probabilities of the presynaptic families








    # Here we collect the information about the target cell morpho-electrophysiology
    node_types = pd.read_csv('./Allen_Inst_V1_model_features/v1_node_types.csv', sep=' ')
    morphology_file = node_types[node_types['node_type_id'] == target_node_id]['morphology'].item()
    dynamics_file = node_types[node_types['node_type_id'] == target_node_id]['dynamics_params'].item()

    morphology_path = join(morphology_dir, morphology_file)

    dynamics_dir = './Allen_Inst_V1_model_features/biophys_components/biophysical_neuron_templates/'
    dynamics_path = join(dynamics_dir, dynamics_file)

    params = json.load(open(dynamics_path, 'r'))

    Vrest = -72.0 # initial guess of membrane potential of target population
    cellParameters = {
        'morphology': morphology_path,
        'templatename': "Biophys1",
        'templatefile': template_file,
        'templateargs': morphology_path,
        'v_init': Vrest,    # initial membrane potential
        'passive': False,   # turn on NEURONs passive mechanism for all sections
        'nsegs_method': 'fixed_length',  # spatial discretization method
        'max_nsegs_length': 20.,
        'tstart': 0,      # start time of simulation, recorders start at t=0
        'pt3d': True,
        'custom_fun': [utils_functions.aibs_perisomatic_NICO],
        'custom_fun_args': [{"dynamics_params": params}],
        'dynamics_path' : dynamics_path
    }



    # Here we build the probe
    somatic_positions = load_dataframe('./Allen_Inst_V1_model_features/somatic_positions.pkl')
    population_params = compute_population_parameters(somatic_positions, target_node_id)

    populationParameters={
            'radius': population_params['radius'],  # population radius (µm)
            'loc': 0.,  # average depth of cell bodies: scaled to 0 for comodity
            'scale': population_params['scale']}  # depth profile of somaitc position alongdepth axis

    probe = GaussCylinderPotential(
        cell=None,
        z=np.linspace(0., -1000., 26)+population_params['center_of_somatic_pos'],  # depth of contacts (µm)
        sigma=0.3,  # tissue conductivity (S/m)
        R=populationParameters['radius'],  #
        sigma_z=population_params['scale'], # this should be the extension of the somatic uniform distribution
        )





    synaptic_positions_V1_to_V1 = pd.read_pickle('./Allen_Inst_V1_model_features/synaptic_positions/V1_to_e23_synaptic_positions.pkl')
    out_of_layer23 = []
    for i,v in synaptic_positions_V1_to_V1.iterrows(): 
        if(v['source'].startswith('e23') or v['source'].startswith('LIFe23') or v['source'].startswith('i23') or v['source'].startswith('LIFi23')): 
            out_of_layer23.append(True)
        else: 
            out_of_layer23.append(False)
    synaptic_positions_V1_to_V1 = synaptic_positions_V1_to_V1[out_of_layer23]
    
    
    
    synaptic_positions_LGN_to_V1 = pd.read_pickle('./Allen_Inst_V1_model_features/synaptic_positions/LGN_to_e23_synaptic_positions.pkl')

    synaptic_positions_BKG_to_V1 = pd.read_pickle('./Allen_Inst_V1_model_features/synaptic_positions/BKG_to_e23_synaptic_positions.pkl')
    synaptic_positions_BKG_to_V1['source'] = 'BKG'

    synaptic_positions_FB_to_V1 = pd.read_pickle('./Allen_Inst_V1_model_features/synaptic_positions/FB_to_e23_synaptic_positions.pkl')
    synaptic_positions_FB_to_V1['source'] = 'FB'





    # Initialize parameter lists
    mult_params = []
    delay_params = []
    synapse_params = []
    synapse_pos_args = []
    
    # Default positional arguments
    default_synapse_pos_args = {
        'section': ['dend', 'apic'],
        'fun': [st.norm],
        'funargs': [{'loc': 0.0, 'scale': 0.01}],
        'funweights': [1.0]
    }
    
    ###########################################################################
    # Process synaptic features and positions within L23
    for _, vector in connectivity_L23V1_to_L23V1.iterrows():
        
    
        mult_params.append(add_mult_params(3, 4))
        delay_params.append(add_delay_params(vector['delay']))
        
        syn_type = get_synapse_type(vector['model_template'])
        params_syn = load_synaptic_params(vector['dynamics_params'])
        

        weight = handle_missing_weight(vector['syn_weight_median'])
        synapse_params.append({'weight': weight, 'syntype': syn_type, **params_syn})
        
        pos_args = process_synaptic_positions(vector, synaptic_positions_V1_to_V1, default_synapse_pos_args)
        synapse_pos_args.append(pos_args if pos_args else default_synapse_pos_args)
    
    ###########################################################################
    # Process synaptic features and positions from LGN to L23
    for _, vector in connectivity_LGN_to_L23V1.iterrows():
        
    
        mult_params.append(add_mult_params(vector['nsyns_from_build'], 0.05))
        delay_params.append(add_delay_params(vector['delay']))
        
        syn_type = get_synapse_type(vector['model_template'])
        params_syn = load_synaptic_params(vector['dynamics_params'])
        

        weight = handle_missing_weight(vector['syn_weight'])
        synapse_params.append({'weight': weight, 'syntype': syn_type, **params_syn})
        
        pos_args = process_synaptic_positions_LGN(vector, synaptic_positions_LGN_to_V1, default_synapse_pos_args)
        synapse_pos_args.append(pos_args if pos_args else default_synapse_pos_args)
    
    ###########################################################################
    # Process synaptic features and positions from BKG to L23
    mult_params.append(add_mult_params(connectivity_BKG_to_L23V1['nsyns_list_from_build'].item(), 0.05))
    delay_params.append(add_delay_params(connectivity_BKG_to_L23V1['delay'].item()))
    
    syn_type = get_synapse_type(connectivity_BKG_to_L23V1['model_template'].item())
    params_syn = load_synaptic_params(connectivity_BKG_to_L23V1['dynamics_params'].item())
    

    weight = handle_missing_weight(connectivity_BKG_to_L23V1['syn_weight'].item())
    synapse_params.append({'weight': weight, 'syntype': syn_type, **params_syn})
    
    pos_args = process_synaptic_positions(connectivity_BKG_to_L23V1.iloc[0], synaptic_positions_BKG_to_V1, default_synapse_pos_args)
    synapse_pos_args.append(pos_args if pos_args else default_synapse_pos_args)

    ###########################################################################
    # Process synaptic features and positions from FB to L23
    mult_params.append(add_mult_params(connectivity_FB_to_L23V1['nsyns_list_from_build'].item(), 0.05))
    delay_params.append(add_delay_params(connectivity_FB_to_L23V1['delay'].item()))
    
    syn_type = get_synapse_type(connectivity_FB_to_L23V1['model_template'].item())
    params_syn = load_synaptic_params(connectivity_FB_to_L23V1['dynamics_params'].item())
    

    weight = handle_missing_weight(connectivity_FB_to_L23V1['syn_weight'].item())
    synapse_params.append({'weight': weight, 'syntype': syn_type, **params_syn})
    
    pos_args = process_synaptic_positions(connectivity_FB_to_L23V1.iloc[0], synaptic_positions_FB_to_V1, default_synapse_pos_args)
    synapse_pos_args.append(pos_args if pos_args else default_synapse_pos_args)
        
        
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    # Let's upload the spiking data of our network: it will be useful to estimate
    # average firing of presynaptic populations    
    
    spikes_tables = pd.read_pickle('./Allen_Inst_V1_model_features/only_L23_spiking_gratings/spike_only_L23_gratings.pkl')
    spikes_tables_LGN = pd.read_pickle('./Allen_Inst_V1_model_features/only_L23_spiking_gratings/spike_lgn_gratings.pkl')
    spikes_tables_BKG = pd.read_pickle('./Allen_Inst_V1_model_features/only_L23_spiking_gratings/spike_bkg_gratings.pkl')
    spikes_tables_FB = pd.read_pickle('./Allen_Inst_V1_model_features/only_L23_spiking_gratings/spike_fb_gratings.pkl')
    
    
    agg_functions = {
        'spike_times': lambda x: sum(x, []),
        'N': 'sum'
    }
    spikes_tables = spikes_tables.groupby('pop_name').agg(agg_functions).reset_index()
        
    spike_times_LGN_cleaned = []
    for i,v in spikes_tables_LGN.iterrows(): 
        lgn_spike_v = v['spike_times']
        spike_times_LGN_cleaned.append(lgn_spike_v[lgn_spike_v<=(simulation_length*1000)]) # spikes time are in ms in bmtk
    spikes_tables_LGN['spike_times'] = spike_times_LGN_cleaned
    
    spikes_tables_BKG = spikes_tables_BKG[spikes_tables_BKG['timestamps']<simulation_length*1000]
    spikes_tables_FB = spikes_tables_FB[spikes_tables_FB['timestamps']<simulation_length*1000]
    
    ###########################################################################
    # We compute now the conductance variation in time to then estimate the various discrete levels of kernels
    dt_approx = 1 # ms -> insteasd of computing the conductance every dt time step we compute it on a coarser time step to enhance efficiency
      
    dizionario_nu = {}
    for idx_sor, sorgenti in enumerate(X):
        if(sorgenti.startswith('s') or sorgenti.startswith('t')):
            _, s_rate = get_spike_rate(list(spikes_tables_LGN[spikes_tables_LGN['pop_name'] == sorgenti]['spike_times']), dt_approx, simulation_length*1000)
            dizionario_nu[sorgenti] = s_rate
            
        elif(sorgenti.startswith('BKG')):
            _, s_rate = get_spike_rate(list(spikes_tables_BKG['timestamps']), dt_approx, simulation_length*1000)
            dizionario_nu[sorgenti] = s_rate 
            
        elif(sorgenti.startswith('FB')): 
            _, s_rate = get_spike_rate(list(spikes_tables_FB['timestamps']), dt_approx, simulation_length*1000)
            dizionario_nu[sorgenti] = s_rate
            
        else: 
            _, s_rate = get_spike_rate(list(spikes_tables[spikes_tables['pop_name'] == sorgenti]['spike_times']), dt_approx, simulation_length*1000)
            dizionario_nu[sorgenti] = s_rate
    
    
    
    kernel_tmp = KernelApprox(
        X=X,
        Y=Y,
        N_X=N_X,
        N_Y=N_Y,
        C_YX=C_YX,
        cellParameters=cellParameters,
        populationParameters=populationParameters,
        #######################################################################
        multapseFunction=st.uniform,
        multapseParameters=mult_params,
        #######################################################################
        delayFunction=st.truncnorm,
        delayParameters=delay_params,
        #######################################################################
        synapseParameters=synapse_params,
        #######################################################################
        synapsePositionArguments = synapse_pos_args,
        #######################################################################
        extSynapseParameters={'syntype': 'Exp2Syn', 'weight': 0, 'tau1': 0.1, 'tau2': 0.11, 'e': 0.0},
        nu_ext=0.,  # external activation rate (spikes/s)
        n_ext=0,  # number of extrinsic synapses
        nu_X=dizionario_nu
        )
    
    
    # here we compute the leak conductance change across simulated time
    g_l_offset = kernel_tmp.get_kernel_effective_conductance(
        probes=[probe],
        Vrest=Vrest, dt=dt_kernels, X=X[0], t_X=t_X, tau=tau,
        g_eff=g_eff, zaxis_rotation = zaxis_rotation)
    
    
    
    
    # here we compute the time index in which the kernels assume the discrete levels of change of leak conductance due to presynaptic activation
    top_percentile = 99
    bottom_percentile = 1
    
    bin_edges = np.linspace(np.percentile(g_l_offset, bottom_percentile), np.percentile(g_l_offset, top_percentile), num_levels + 1)
    g_l_offset[g_l_offset>np.percentile(g_l_offset, top_percentile)]  = np.percentile(g_l_offset, top_percentile)
    g_l_offset[g_l_offset<np.percentile(g_l_offset, bottom_percentile)]  = np.percentile(g_l_offset, bottom_percentile)
    time_series_discretized = pd.cut(g_l_offset.flatten(), bins=bin_edges, labels=False, include_lowest=True)
    time_series_discretized_new = np.repeat(time_series_discretized, int(dt_approx / dt_simulation))
    
    del kernel_tmp
    
    
    kernel_dictionary = {} # this dictionary will contains all the info for computing the LFP-kernels for each discrete level of change of leak conductance
    for discrete_lvl_index in range(num_levels): 
    
        dizionario_nu = {}
        for idx_sor, sorgenti in enumerate(X):
    
            if(sorgenti.startswith('s') or sorgenti.startswith('t')):
                _, s_rate = get_spike_rate(list(spikes_tables_LGN[spikes_tables_LGN['pop_name'] == sorgenti]['spike_times']), dt_simulation, simulation_length*1000)
                dizionario_nu[sorgenti] = sum(s_rate[time_series_discretized_new == discrete_lvl_index])/N_X[idx_sor]/(sum(time_series_discretized==discrete_lvl_index)*dt_approx/1000)
                
            elif(sorgenti.startswith('BKG')):
                _, s_rate = get_spike_rate(list(spikes_tables_BKG['timestamps']), dt_simulation, simulation_length*1000)
                dizionario_nu[sorgenti] = sum(s_rate[time_series_discretized_new == discrete_lvl_index])/N_X[idx_sor]/(sum(time_series_discretized==discrete_lvl_index)*dt_approx/1000)
                
            elif(sorgenti.startswith('FB')): 
                _, s_rate = get_spike_rate(list(spikes_tables_FB['timestamps']),dt_simulation, simulation_length*1000)
                dizionario_nu[sorgenti] = sum(s_rate[time_series_discretized_new == discrete_lvl_index])/N_X[idx_sor]/(sum(time_series_discretized==discrete_lvl_index)*dt_approx/1000)
                
            else: 
                _, s_rate = get_spike_rate(list(spikes_tables[spikes_tables['pop_name'] == sorgenti]['spike_times']), dt_simulation, simulation_length*1000)
                dizionario_nu[sorgenti] = sum(s_rate[time_series_discretized_new == discrete_lvl_index])/N_X[idx_sor]/(sum(time_series_discretized==discrete_lvl_index)*dt_approx/1000)
    
            kernel = KernelApprox(
                X=X,
                Y=Y,
                N_X=N_X,
                N_Y=N_Y,
                C_YX=C_YX,
                cellParameters=cellParameters,
                populationParameters=populationParameters,
                #######################################################################
                multapseFunction=st.uniform,
                multapseParameters=mult_params,
                #######################################################################
                delayFunction=st.truncnorm,
                delayParameters=delay_params,
                #######################################################################
                synapseParameters=synapse_params,
                #######################################################################
                synapsePositionArguments = synapse_pos_args,
                #######################################################################
                extSynapseParameters={'syntype': 'Exp2Syn', 'weight': 0, 'tau1': 0.1, 'tau2': 0.11, 'e': 0.0},
                nu_ext=0.,  # external activation rate (spikes/s)
                n_ext=0,  # number of extrinsic synapses
                nu_X=dizionario_nu
                )
            
            
            kernel_dictionary[discrete_lvl_index] = kernel    
    
    return X, Y, kernel_dictionary, probe, Vrest, dt_kernels, t_X, tau, g_eff, time_series_discretized_new









#%% this loop compute the kernels by launching the MC simulation of a single target neuron

def compute_LFP_bmtk_to_population_fromV1(target_population, target_query_node_type_id, zaxis_rotation, num_levels):

    simulation_length = 3 # seconds of simulations when V1 was presented with gratings
    dt_simulation = 0.05 # ms: discrete time steps used in bmtk simulations
    num_levels = 2 # number of discrete levels in which discretizing the leak conductance changes
    X, Y, kernel_dict, probe, Vrest, dt, t_X, tau, g_eff, time_series_discretized = compute_utils_from_bmtk_to_population(target_population, target_query_node_type_id, simulation_length, dt_simulation, zaxis_rotation, num_levels)
    
    save_folder = './kernels/gratings/'

    save_folder += target_population+'/'+str(target_query_node_type_id)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    open_file = open(save_folder+'/vector_time_normalized_levels', "wb")
    pickle.dump(time_series_discretized, open_file)
    open_file.close()
    
    
    
    for numero_livelli in range(num_levels): 
        folder_save = save_folder+'/'+str(numero_livelli)
    
        if not os.path.exists(folder_save):
            os.makedirs(folder_save)
    
        loop_iter = 0
        for source_presynaptic in X:
            loop_iter += 1
            print('num_levels: '+str(numero_livelli+1)+' of '+str(num_levels)+': processing from ' + source_presynaptic +' to ' + Y +' %d of %d' %(loop_iter,len(X)))
    
            if source_presynaptic not in os.listdir(folder_save):
                kernel = kernel_dict[numero_livelli]
                
                H = kernel.get_kernel(
                    probes=[probe],
                    Vrest=Vrest, dt=dt, X=source_presynaptic, t_X=t_X, tau=tau,
                    g_eff=g_eff, zaxis_rotation = zaxis_rotation)
    
                dict_kernels = {}
                dict_kernels['source'] = source_presynaptic
                dict_kernels['target'] = Y
                dict_kernels['kernel'] = H

            open_file = open(folder_save+'/'+source_presynaptic, "wb")
            pickle.dump(dict_kernels, open_file)
            open_file.close()






###############################################################################
#%% MAIN
connectivity_V1_V1 = pd.read_pickle('./Allen_Inst_V1_model_features/features_V1_V1_L23.pkl')

string_target_vector = ['e23Cux2'] # this code is meant to estimate LFP generated by pyramidal cells in L23
num_levels = 2 # number of discrete levels to estimate the change in leak conductance due to persistent presynptic activation


for index_target in range(len(string_target_vector)):
    string_target = string_target_vector[index_target]


    if(string_target.startswith('e1') or string_target.startswith('i1')):
        layer = 'VisL1'
    elif(string_target.startswith('e23') or string_target.startswith('i23')):
        layer = 'VisL23'
    elif(string_target.startswith('e4') or string_target.startswith('i4')):
        layer = 'VisL4'
    elif(string_target.startswith('e5') or string_target.startswith('i5')):
        layer = 'VisL5'
    elif(string_target.startswith('e6') or string_target.startswith('i6')):
        layer = 'VisL6'


    rotazioni_info = json.load(open('./Allen_Inst_V1_model_features/v1_node_models.json', 'r'))
    rotazioni_info = rotazioni_info['locations'][layer][string_target]['models']


    query_node_type_id = np.unique(connectivity_V1_V1 [connectivity_V1_V1['target_query'] == string_target]['target_query_node_type_id'])

    alpha_loop = 0
    for nodi in query_node_type_id:
        alpha_loop+=1
        print('processing nodo %d of %d' %(alpha_loop,len(query_node_type_id)))

        ind1 = connectivity_V1_V1['target_query'] == string_target
        ind2 = connectivity_V1_V1['target_query_node_type_id'] == nodi


        for ind_rotation in rotazioni_info:
            if(ind_rotation['node_type_id'] == nodi):
                angolo_rotazione = ind_rotation.get('rotation_angle_zaxis', 0.0)


        if(list(connectivity_V1_V1[np.logical_and(ind1,ind2)]['numerosity_target_nodes'])[0]>100):
            compute_LFP_bmtk_to_population_fromV1(string_target, nodi, angolo_rotazione, num_levels)







