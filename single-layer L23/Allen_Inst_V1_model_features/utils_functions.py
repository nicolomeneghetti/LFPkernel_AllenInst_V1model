import os
from os.path import join
import json
import numpy as np
import matplotlib.pyplot as plt
import neuron
from neuron import h
import LFPy
import h5py
import bmtk
import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np
import neuron
import pandas as pd
from typing import Dict, List, Union


def initial_setups():
    # recompile mod files if needed
    mech_loaded = neuron.load_mechanisms('mod')
    if not mech_loaded:
        os.system('cd mod && nrnivmodl && cd -')
        mech_loaded = neuron.load_mechanisms('mod')
    print(f'mechanisms loaded: {mech_loaded}')
    
    
    # These paths might have to be updated to point towards the needed files:
    bmtk_dir = bmtk.__path__[0]
    template_dir = join(bmtk_dir, "simulator", "bionet", "default_templates")
    template_file = join(template_dir, "Biophys1.hoc")
    
    bio_components_dir = './biophys_components'
    mechanisms_dir = join(bio_components_dir, "mechanisms")
    morphology_dir = join(bio_components_dir, "morphologies")
    dynamics_dir = join(bio_components_dir, "biophysical_neuron_templates", "ctdb")
        
    print(mechanisms_dir)
    # Requires that the mechanisms in this folder is compiled (run 'nrnivmodl' in a terminal in the folder):
    neuron.load_mechanisms(mechanisms_dir)
    
    
    return morphology_dir, dynamics_dir, template_file
    
        
def fix_axon_peri(cell):
    """Replace reconstructed axon with a stub
    This function is extracted from bmtk.simulator.bionet.default_setters.cell_models
    :param hobj: hoc object
    """
    hobj = cell.template
    for sec in h.allsec():
        if "axon" in sec.name():
            h.delete_section(sec=sec)

    h.execute('create axon[2]', hobj)

    for sec in h.allsec():
        if "axon" in sec.name():
            sec.L = 30
            sec.diam = 1
            hobj.axonal.append(sec=sec)
            hobj.all.append(sec=sec)
    #
    hobj.axon[0].connect(hobj.soma[0], 0.5, 0)
    hobj.axon[1].connect(hobj.axon[0], 1, 0)
    h.define_shape()


def set_params_peri(cell, biophys_params):
    """Set biophysical parameters for the cell
    This function is extracted from bmtk.simulator.bionet.default_setters.cell_models

    :param hobj: NEURON's cell object
    :param biophys_params: name of json file with biophys params for cell's model which determine spiking behavior
    :return:
    """
    hobj = cell.template
    passive = biophys_params['passive'][0]
    conditions = biophys_params['conditions'][0]
    genome = biophys_params['genome']

    # Set passive properties
    cm_dict = dict([(c['section'], c['cm']) for c in passive['cm']])
    for sec in hobj.all:
        sec.Ra = passive['ra']
        sec.cm = cm_dict[sec.name().split(".")[1][:4]]
        sec.insert('pas')

        for seg in sec:
            seg.pas.e = passive["e_pas"]

    # Insert channels and set parameters
    for p in genome:
        sections = [s for s in hobj.all if s.name().split(".")[1][:4] == p["section"]]

        for sec in sections:
            if p["mechanism"] != "":
                sec.insert(p["mechanism"])
            setattr(sec, p["name"], p["value"])

    # Set reversal potentials
    for erev in conditions['erev']:
        sections = [s for s in hobj.all if s.name().split(".")[1][:4] == erev["section"]]
        for sec in sections:
            sec.ena = erev["ena"]
            sec.ek = erev["ek"]


# new function that calls aibs perisomatic and then it runs the make passive stuff 
def aibs_perisomatic_NICO(cell, dynamics_params):
    """This function is extracted from bmtk.simulator.bionet.default_setters.cell_models"""
    if dynamics_params is not None:
        fix_axon_peri(cell)
        set_params_peri(cell, dynamics_params)
    
    remove_list = []
    for p in dynamics_params['genome']:
        remove_list.append(p["mechanism"])

    mt = h.MechanismType(0)
    for sec in h.allsec():
        for seg in sec:
            for mech in remove_list:
                mt.select(mech)
                mt.remove(sec=sec)
                
        
    return cell





def load_dataframe(filepath: str) -> pd.DataFrame:
    """Loads a DataFrame from a pickle file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    return pd.read_pickle(filepath)


def filter_connectivity(df: pd.DataFrame, target_population: str, target_node_id: int) -> pd.DataFrame:
    """Filters connectivity data based on population and node type."""
    is_target_pop = df['target_query'] == target_population
    is_target_node = df['target_query_node_type_id'] == target_node_id
    try: 
        valid_weights = df['syn_weight_median'] > 0
    except: 
        valid_weights = df['syn_weight'] > 0
        
    valid_probabilities = df['conn_probability'] > 0
    return df[is_target_pop & is_target_node & valid_weights & valid_probabilities]

def filter_layer23(df: pd.DataFrame) -> pd.DataFrame:
    """Filters rows where the source is from layer 23."""
    is_layer23 = df['source_query'].str.startswith(('e23', 'LIFe23', 'i23', 'LIFi23'))
    return df[is_layer23]


def get_population_size(df: pd.DataFrame, column: str) -> Union[int, float]:
    """Extracts the population size from a DataFrame column."""
    if df.empty:
        return 0
    return int(df[column].iloc[0])


def parse_node_types(filepath: str, target_node_id: int) -> Dict[str, str]:
    """Parses node type information to get morphology and dynamics."""
    node_types = pd.read_csv(filepath, sep=' ')
    target_row = node_types[node_types['node_type_id'] == target_node_id]
    if target_row.empty:
        raise ValueError(f"No matching node_type_id {target_node_id} found in {filepath}")
    return {
        "morphology": target_row['morphology'].item(),
        "dynamics_params": target_row['dynamics_params'].item(),
    }


def load_synaptic_params(filepath: str) -> Dict:
    """Loads synaptic parameters from a JSON file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Synaptic params file not found: {filepath}")
    return json.load(open(filepath, 'r'))


def process_synapse_positions(df: pd.DataFrame, target_query: str, source_query: str) -> Dict:
    """Processes synapse position arguments for a given target-source pair."""
    filtered = df[(df['source'] == source_query) & (df['target'] == target_query)]
    if filtered.empty or filtered['num_sinapsi'].item() <= 20:
        # Default synapse distribution for low synapse count
        return {
            'section': ['dend', 'apic'],
            'fun': [st.norm],
            'funargs': [{'loc': 0.0, 'scale': 0.01}],
            'funweights': [1.0],
        }

    # Complex distribution for higher synapse counts
    weights = [d.pop('funweights') for sublist in filtered['node_ids'] for d in sublist]
    return {
        'section': ['dend', 'apic'],
        'fun': [st.norm] * len(weights),
        'funargs': filtered['node_ids'].item(),
        'funweights': weights,
    }


def compute_population_parameters(posizione_somatiche: pd.DataFrame, target_node_id: int) -> Dict:
    """Computes population parameters such as radius and depth range."""
    target_row = posizione_somatiche[posizione_somatiche['node_type_id'] == target_node_id]
    if target_row.empty:
        raise ValueError(f"No matching node_type_id {target_node_id} found in somatic positions")
    radial_range = target_row['radial_range'].item()[1]
    depth_range = target_row['depth_range'].item()
    return {
        "radius": radial_range,
        "loc": 0.0,
        "scale": depth_range[1] - depth_range[0],
        "center_of_somatic_pos": (depth_range[1] + depth_range[0])/2,
    }



def get_spike_rate(times, dt = 2**-7, durata_simulazioni=3*1000):
    bins = (np.arange(0 / dt, durata_simulazioni / dt + 1)
            * dt - dt / 2)
    hist, _ = np.histogram(times, bins=bins)
    return bins, hist.astype(float)

def add_mult_params(loc, scale):
    """Add multiplicative parameters."""
    return {'loc': loc, 'scale': scale}

def add_delay_params(delay):
    """Add delay parameters."""
    return {'a': delay - 0.1, 'b': delay + 0.1, 'loc': delay, 'scale': 0.05}

def get_synapse_type(model_template):
    """Determine synapse type."""
    return 'Exp2Syn' if model_template == 'exp2syn' else model_template

def load_synaptic_params(dynamics_params, synaptic_dir='./Allen_Inst_V1_model_features/biophys_components/synaptic_models/'):
    """Load synaptic dynamics parameters."""
    synaptic_path = join(synaptic_dir, dynamics_params)
    return json.load(open(synaptic_path, 'r'))

def handle_missing_weight(weight):
    """Handle missing weights by returning 0 if NaN."""
    return 0.0 if np.isnan(weight) else weight

def process_synaptic_positions(vector, synaptic_data, default_args):
    """Process synaptic position arguments."""
    condition = (
        (synaptic_data['source'] == vector['source_query']) &
        (synaptic_data['target'] == vector['target_query'])
    )
    synapse_data = synaptic_data[condition].copy()

    if sum(condition) and synapse_data['num_sinapsi'].item() > 20:

        funweights_list = [d.pop('funweights') for sublist in synapse_data['node_ids'] for d in sublist]
        
        
        target_sections = vector['target_sections']
        replacements = {'[': '', ']': '', '"': '', 'apical': 'apic', 'basal': 'dend', 'somatic': 'soma'}
        for old, new in replacements.items():
            target_sections = target_sections.replace(old, new)
        sections = target_sections.split(", ")
        return {
            'section': sections,
            'fun': [st.norm] * len(funweights_list),
            'funargs': synapse_data['node_ids'].item(),
            'funweights': funweights_list
        }
    return default_args

def process_synaptic_positions_LGN(vector, synaptic_data, default_args):
    """Process synaptic position arguments."""
    condition = (
        (synaptic_data['source'] == vector['LGN_pop_name']) &
        (synaptic_data['target'] == vector['target_query'])
    )
    synapse_data = synaptic_data[condition].copy()

    if sum(condition) and synapse_data['num_sinapsi'].item() > 20:

        funweights_list = [d.pop('funweights') for sublist in synapse_data['node_ids'] for d in sublist]
        
        
        target_sections = vector['target_sections']
        replacements = {'[': '', ']': '', '"': '', 'apical': 'apic', 'basal': 'dend', 'somatic': 'soma'}
        for old, new in replacements.items():
            target_sections = target_sections.replace(old, new)
        sections = target_sections.split(", ")
        return {
            'section': sections,
            'fun': [st.norm] * len(funweights_list),
            'funargs': synapse_data['node_ids'].item(),
            'funweights': funweights_list
        }
    return default_args





def append_default_synapse_pos_args(synapse_pos_args):
    """Append default synapse positional arguments."""
    synapse_pos_args.append({
        'section': ['dend', 'apic'],
        'fun': [st.norm],
        'funargs': [{'loc': 0.0, 'scale': 0.01}],
        'funweights': [1.0]
    })