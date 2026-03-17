import os, shutil
import pickle
import traceback
import copy
import math
from dataclasses import dataclass
from dataclasses import field
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from bayesian_torch.utils.util import predictive_entropy, mutual_information
import scipy.stats as stats
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss
from botorch.acquisition import qExpectedImprovement
from botorch.acquisition import qUpperConfidenceBound
from botorch.acquisition import UpperConfidenceBound
from botorch.fit import fit_gpytorch_model
from botorch.generation import MaxPosteriorSampling
from botorch.models import FixedNoiseGP, SingleTaskGP, HeteroskedasticSingleTaskGP
from botorch.optim import optimize_acqf, optimize_acqf_discrete,optimize_acqf_mixed
from botorch.utils.transforms import normalize
from botorch.utils.transforms import unnormalize
from torch.quasirandom import SobolEngine
import subprocess
import re
import gpytorch
from gpytorch.constraints import Interval
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import HorseshoePrior
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel, RBFKernelGrad, SpectralDeltaKernel, SpectralMixtureKernel
from typing import List, Optional, Tuple
from botorch.test_functions.base import BaseTestProblem
from torch import Tensor
import time
from __future__ import annotations
import psutil
from copy import deepcopy
from typing import Dict, Optional, Tuple, Union
from botorch.test_functions.synthetic import SyntheticTestFunction
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.exceptions import UnsupportedError
from botorch.models.gp_regression_mixed import MixedSingleTaskGP
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.model import Model
from botorch.posteriors.posterior import Posterior
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import convert_to_target_pre_hook, t_batch_mode_transform
from torch.distributions import Normal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import drprobe as drp
import itertools
import IPython.display as display
import time

torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double
print(device)
SMOKE_TEST = os.environ.get("SMOKE_TEST")
root_path = os.getcwd()+'/'

def BuildCell(output_file, cif=None, spacegroup=None, lattice=None, atoms=None, symmetry=None, override=False, output=False):
    
    _command = "BuildCell --output {}".format(output_file)

    if cif is not None:
        _command += ' --cif={}'.format(cif)
        
    if spacegroup is not None:
        _command += ' --spacegroup={}'.format(spacegroup)
        
    if lattice is not None:
        values = [str(value) for value in lattice]
        result_string = ','.join(values)
        _command += ' --lattice={}'.format(result_string)
        
    if atoms is not None:
          for element, properties in atoms.items():
            if isinstance(properties, tuple):
                properties_str = ','.join(map(str, properties))
                _command += ' --atom={},{}'.format(element, properties_str)
            elif isinstance(properties, list):
                for atom_properties in properties:
                    atom_properties_str = ','.join(map(str, atom_properties))
                    _command += ' --atom={},{}'.format(element, atom_properties_str)
            else:
                properties_str = str(properties)
                _command += ' --atom={},{}'.format(element, properties_str)

    if symmetry is not None:
        _command += ' --symmetry={}'.format(symmetry)

    if override:
        _command += ' --override'

    if output:
        co = subprocess.check_output('/home/phD/xiankang/anaconda3/envs/drprobe/lib/python3.11/site-packages/drprobe/'+_command, shell=True)  ## replace the path "/home/phD/xiankang/anaconda3/envs/drprobe/lib/python3.11/site-packages/drprobe/'" with yours where the drprobe is installed.
        print('Performed BuildCell with the following command:\n','/home/phD/xiankang/anaconda3/envs/drprobe/lib/python3.11/site-packages/drprobe/' +_command)
        print(co.decode('utf-8'))
    else:
        subprocess.call(_command, shell=True)

def slice_dict_by_z(data):
    # Collect all z values
    z_values = [tup[2] for key in data for tup in data[key]]

    # Calculate the boundaries for 4 slices
    z_min, z_max = min(z_values), max(z_values)
    z_slices = np.linspace(z_min, z_max, 5)  # 5 points divide into 4 intervals

    # Initialize 4 separate dictionaries for the slices
    slice1, slice2, slice3, slice4 = {}, {}, {}, {}

    # Distribute tuples into the slices
    for key in data:
        for tup in data[key]:
            z = tup[2]
            if z_slices[0] <= z < z_slices[1]:
                if key not in slice1:
                    slice1[key] = []
                slice1[key].append(tup)
            elif z_slices[1] <= z < z_slices[2]:
                if key not in slice2:
                    slice2[key] = []
                slice2[key].append(tup)
            elif z_slices[2] <= z < z_slices[3]:
                if key not in slice3:
                    slice3[key] = []
                slice3[key].append(tup)
            else:
                if key not in slice4:
                    slice4[key] = []
                slice4[key].append(tup)

    return slice1, slice2, slice3, slice4

def decimal_to_binary(decimal):
    binary_list = []
    n = 20

    # Convert decimal number to binary representation
    while decimal > 0:
        binary_list.insert(0, decimal % 2)
        decimal //= 2

    # Pad the binary list to ensure it has exactly 20 elements
    while len(binary_list) < n:
        binary_list.insert(0, 0)

    return binary_list

def recover_dict_from_slices(slice1, slice2, slice3, slice4):
    # Initialize an empty dictionary for recovery
    recovered_dict = {}

    # Combine the contents of all four slices
    for slice_dict in (slice1, slice2, slice3, slice4):
        for key, tuples in slice_dict.items():
            if key not in recovered_dict:
                recovered_dict[key] = []
            recovered_dict[key].extend(tuples)

    return recovered_dict

def sample_z_from_theta(theta, num_samples=1):
    theta = torch.clamp(theta, 0, 1)  # Ensure theta is a valid probability
    samples = torch.bernoulli(theta.repeat(num_samples, 1))
    return samples

def renew_atoms(atoms_dict, new_list):
    i = 0
    for key, values in atoms_dict.items():
        for index, value in enumerate(values):
            atoms_dict[key][index] = value[:3] + (new_list[i],) + value[4:]
            i += 1
    return atoms_dict

def renew_atoms_2(atoms_dict, new_list):
    i = 0
    for key, values in atoms_dict.items():
        for index, value in enumerate(values):
            atoms_dict[key][index] = value[:4] + (new_list[i],) 
            i += 1
    return atoms_dict

def save_msa_prm(prm_filename, msa_parameters, random_slices=False):
    directory = os.path.split(prm_filename)[0]
    if directory:
        if not os.path.isdir(directory):
            os.makedirs(directory, exist_ok=True)

    aberrations = {0: 'image_shift',
                   1: 'defocus',
                   2: '2-fold-astigmatism',
                   3: 'coma',
                   4: '3-fold-astigmatism',
                   5: 'CS',
                   6: 'star_aberration',
                   7: '4-fold-astigmatism',
                   8: 'coma(5th)',
                   9: 'lobe-aberration',
                   10: '5-fold-astigmatism',
                   11: 'C5'}

    with open(prm_filename, 'w') as prm:
        prm.write("'[Microscope Parameters]'\n")
        prm.write("{} ! Semi angle of convergence [mrad]\n".format(msa_parameters['conv_semi_angle']))
        prm.write("{} ! Inner radius of the annular detector [mrad]\n".format(msa_parameters['inner_radius_ann_det']))
        prm.write("{} ! Outer radius of the annular detector [mrad]\n".format(msa_parameters['outer_radius_ann_det']))
        prm.write("{}, '{}' ! Detector definition file, switch and file name\n".format(msa_parameters['detector'][0], msa_parameters['detector'][1]))
        prm.write("{} ! Electron wavelength [nm]\n".format(msa_parameters['wavelength']))
        prm.write("{} ! De-magnified source radius (1/e half width) [nm]\n".format(msa_parameters['source_radius']))
        prm.write("{} ! Focus spread [nm]\n".format(msa_parameters['focus_spread']))
        prm.write("{} ! Focus-spread kernel half-width w.r.t. the focus spread\n".format(msa_parameters['focus_spread_kernel_hw']))
        prm.write("{} ! Focus-spread kernel size\n".format(msa_parameters['focus_spread_kernel_size']))
        prm.write("{} ! Number of aberration definitions following\n".format(msa_parameters['number_of_aberrations']))
        for key in msa_parameters['aberrations_dict']:
            prm.write('{} {:.4f} {:.4f} ! {}\n'.format(key, msa_parameters['aberrations_dict'][key][0],
                                                 msa_parameters['aberrations_dict'][key][1],
                                                 aberrations[key]))
        prm.write("'[Multislice Parameters]'\n")
        prm.write("{} ! Object tilt X [deg]. Approximative approach. Do not use for tilts larger than 5 degrees.\n".format(msa_parameters['tilt_x']))
        prm.write("{} ! Object tilt Y [deg]. Approximative approach. Do not use for tilts larger than 5 degrees.\n".format(msa_parameters['tilt_y']))
        prm.write("{} ! Horizontal scan frame offset [nm].\n".format(msa_parameters['h_scan_offset']))
        prm.write("{} ! Vertical scan frame offset [nm].\n".format(msa_parameters['v_scan_offset']))
        prm.write("{} ! Horizontal scan frame size [nm].\n".format(msa_parameters['h_scan_frame_size']))
        prm.write("{} ! Vertical scan frame size [nm].\n".format(msa_parameters['v_scan_frame_size']))
        prm.write("{} ! Scan frame rotation [deg] w.r.t. the slice data.\n".format(msa_parameters['scan_frame_rot']))
        prm.write("{} ! Number of scan columns = number of pixels on horizontal scan image axis.\n".format(msa_parameters['scan_columns']))
        prm.write("{} ! Number of scan rows = number of pixels on vertical scan image axis.\n".format(msa_parameters['scan_rows']))
        prm.write("{} ! Switch for partial temporal coherence calculation. Drastic increase of calculation time if activated.\n".format(msa_parameters['temp_coherence_flag']))
        prm.write("{} ! Switch for partial spatial coherence calculation. Is only applied in combination with an input image file.\n".format(msa_parameters['spat_coherence_flag']))
        prm.write("{} ! Supercell repeat factor in horizontal direction, x.\n".format(msa_parameters['super_cell_x']))
        prm.write("{} ! Supercell repeat factor in vertical direction, y.\n".format(msa_parameters['super_cell_y']))
        prm.write("{} ! Supercell repeat factor in Z-direction, obsolete.\n".format(msa_parameters['super_cell_z']))
        prm.write("{} ! Slice file series name [SFN]. Expected file names are [SFN]+'_###.sli' where ### is a three digit number.\n".format(msa_parameters['slice_files']))
        prm.write("{} ! Number of slice files to load.\n".format(msa_parameters['number_of_slices']))
        prm.write("{} ! Number of frozen lattice variants per slice.\n".format(msa_parameters['number_frozen_lattice']))
        prm.write("{} ! Minimum number of frozen lattice variations averaged per scan pixel in STEM mode.\n".format(msa_parameters['min_num_frozen']))
        prm.write("{} ! Detector readout period in slices.\n".format(msa_parameters['det_readout_period']))
        prm.write("{} ! Number of slices in the object.\n".format(msa_parameters['tot_number_of_slices']))

        if 'slice_ids' in msa_parameters:
            for slice_id in msa_parameters['slice_ids']:
                prm.write("{} ! Slice ID\n".format(slice_id))
        elif random_slices:
            if msa_parameters['tot_number_of_slices'] < msa_parameters['number_of_slices']:
                ld = int(msa_parameters['number_of_slices'] - msa_parameters['tot_number_of_slices'])
                lo = np.random.randint(0, ld)
                for i in range(lo, msa_parameters['tot_number_of_slices'] + lo):
                    prm.write("{} ! Slice ID\n".format(i % msa_parameters['number_of_slices']))
            elif msa_parameters['tot_number_of_slices'] >= msa_parameters['number_of_slices']:
                fm = factors(msa_parameters['tot_number_of_slices'])
                idx = int((len(fm) + 1) / 2)
                len_div = int(fm[idx])
                num_div = int(msa_parameters['tot_number_of_slices'] / len_div)
                ld = msa_parameters['number_of_slices'] - len_div
                for j in range(num_div):
                    lo = np.random.randint(0, ld)
                    for i in range(lo, len_div + lo):
                        prm.write("{} ! Slice ID\n".format(i % msa_parameters['number_of_slices']))
        else:
            for i in range(msa_parameters['tot_number_of_slices']):
                prm.write("{} ! Slice ID\n".format(i % msa_parameters['number_of_slices']))
        prm.write("End of parameter file.")

    # Sort prm file
    with open(prm_filename, 'r+') as prm:
        content = prm.readlines()

        length = []
        for line in content:
            length.append(len(line.split('!')[0]))

        spacer = np.max(length)
        prm.seek(0)
        prm.truncate()
        for line in content:
            if '!' in line:
                line = line.split('!')[0].ljust(spacer) + ' !' + line.split('!')[1]
            prm.write(line)

def load_mas(prm_filename):
    msa_dict = {}
    aberrations_dict = {}
    silce_id = []

    with open(prm_filename, 'r') as prm:
        content = prm.readlines()
        content = [re.split(r'[,\s]\s*', line) for line in content]
        if content[1].index('!') == 3:
            msa_dict['conv_semi_angle'] = (float(content[1][0]), float(content[1][1]), float(content[1][2]))
        elif content[1].index('!') == 1:
            msa_dict['conv_semi_angle'] = float(content[1][0])
        else:
            msa_dict['conv_semi_angle'] = 0
        msa_dict['inner_radius_ann_det'] = float(content[2][0])
        msa_dict['outer_radius_ann_det'] = float(content[3][0])
        msa_dict['detector'] = (int(content[4][0]), content[4][1])
        msa_dict['wavelength'] = float(content[5][0])
        msa_dict['source_radius'] = float(content[6][0])
        msa_dict['focus_spread'] = float(content[7][0])
        msa_dict['focus_spread_kernel_hw'] = float(content[8][0])
        msa_dict['focus_spread_kernel_size'] = float(content[9][0])
        msa_dict['number_of_aberrations'] = int(content[10][0])
        n = int(content[10][0])
        for i in range(n):
            index = int(content[11 + i][0])
            aberrations_dict[index] = (float(content[11 + i][1]), float(content[11 + i][2]))
        msa_dict['aberrations_dict'] = aberrations_dict
        msa_dict['tilt_x'] = float(content[12+n][0])
        msa_dict['tilt_y'] = float(content[13+n][0])
        msa_dict['h_scan_offset'] = float(content[14+n][0])
        msa_dict['v_scan_offset'] = float(content[15+n][0])
        msa_dict['h_scan_frame_size'] = float(content[16+n][0])
        msa_dict['v_scan_frame_size'] = float(content[17+n][0])
        msa_dict['scan_frame_rot'] = float(content[18+n][0])
        msa_dict['scan_columns'] = int(content[19+n][0])
        msa_dict['scan_rows'] = int(content[20+n][0])
        msa_dict['temp_coherence_flag'] = int(content[21+n][0])
        msa_dict['spat_coherence_flag'] = int(content[22+n][0])
        msa_dict['super_cell_x'] = int(content[23+n][0])
        msa_dict['super_cell_y'] = int(content[24+n][0])
        msa_dict['super_cell_z'] = int(content[25+n][0])
        msa_dict['slice_files'] = content[26+n][0]
        msa_dict['number_of_slices'] = int(content[27+n][0])
        msa_dict['number_frozen_lattice'] = int(content[28+n][0])
        msa_dict['min_num_frozen'] = int(content[29+n][0])
        msa_dict['det_readout_period'] = int(content[30+n][0])
        msa_dict['tot_number_of_slices'] = int(content[31+n][0])
        
        if len(content) >= 32 + n:
            m = int(content[31+n][0])
            slice_ids = []
            for j in range(m):
                slice_ids.append(int(content[32+j][0]))
            msa_dict['slice_ids'] = slice_ids
           
    return msa_dict

a, b, c = np.genfromtxt('time/BTO110-2M.cel', skip_header=1, skip_footer=1, usecols=(1, 2, 3))[0]
lattice = (10*a,10*b,10*c, 90.00000, 90.00000, 90.00000)
print(a,b,c)

class IO(object):
    
    def __init__(self, root_path, dr_input_file = 'BTO110-2.cel', dr_input_para_file = 'inputpara.txt', dr_output_para_file = 'outputpara.txt',
                 dr_output_file_name='out.dat', real_data = 'H3.dat'):
    
        self.root_path = root_path
        self.dr_input_file = dr_input_file
        self.dr_input_para_file = dr_input_para_file
        self.dr_output_para_file = dr_output_para_file
        self.dr_output_file_name = dr_output_file_name
        self.real_data = real_data
       
    def read_real_data(self,folder,x,y):
        img_real = np.fromfile(self.root_path+folder+'/'+self.real_data, dtype='float32').reshape((x, y))
        return img_real
    
    def read_input_file(self,folder):
        return ( self.root_path+folder+'/'+self.dr_input_file)
    
    def read_parameters(self, folder):
        with open(self.root_path+folder+'/'+self.dr_input_para_file, 'r') as f:
            inl = f.readlines()
        para_dict = {}    
        for line in inl:
            parts = line.split(': ')
            if len(parts) == 2:
                key = parts[0].strip("'")  # Remove single quotes from the parameter name
                value = parts[1].strip()  # Remove any leading or trailing whitespace
                if value.startswith('(') and value.endswith(')'):
                    # If the value is a tuple, parse it as a tuple
                    value = tuple(map(float, value[1:-1].split(', ')))
                else:
                    value = float(value)  # Convert value to float if not a tuple
                para_dict[key] = value
        return para_dict
    
    def save_parameters_to_file(self, parameters, folder):
        with open(self.root_path+folder+'/'+self.dr_output_para_file, 'w') as f:
            for key, value in parameters.items():
                if isinstance(value, tuple):
                    # If value is a tuple, format it accordingly
                    # formatted_value = f'({", ".join(str(v) for v in value)})'
                    formatted_value = ', '.join(str(v) for v in value)
                else:
                    formatted_value = str(value)

                f.write(f"'{key}' : {formatted_value}\n")

folder = 'time'
io = IO(root_path)
input_file = io.read_input_file(folder)
a, b, c = np.genfromtxt(input_file, skip_header=1, skip_footer=1, usecols=(1, 2, 3))[0]
nx, ny = 128, 90
nz = 4
lattice = (10*a,10*b,10*c, 90.00000, 90.00000, 90.00000)
img_real=np.fromfile('time/ROIs-1.dat', dtype='float32').reshape((270, 256))
kl = img_real[45:90,64:128]
plt.imshow(kl)
plt.colorbar()
plt.show()
y_real_p = torch.tensor(kl, device = device, dtype = dtype)

def calculate_total_occupancies(atoms_list):
    total_occupancies = [0] * 28   #16 slices  
    for i, atoms in enumerate(atoms_list):
        for atom_type, positions in atoms.items():
            for pos in positions:
                z = pos[2]
                occupancy = pos[3]
                layer = int(z * 4)  
                total_occupancies[i * 4 + layer] += occupancy
    # print('total_occupancies:',total_occupancies)
    return total_occupancies

def filter_layers(atoms_list, new_slice_ids):
    # print('atoms_list:', atoms_list)
    total_occupancies = calculate_total_occupancies(atoms_list)
    # print('total_occupancies:', total_occupancies)
    # a_list = new_slice_ids[:12] + new_slice_ids[-12:]
    a_list = new_slice_ids
    # print(a_list)
    # while len(a_list) > 0 and total_occupancies[a_list[0]-4] == 0:
    #     a_list.pop(0)
    # while len(a_list) > 0 and total_occupancies[a_list[-1]-4] == 0:
    #     a_list.pop()
    while len(total_occupancies) > 0 and total_occupancies[0] == 0:
        total_occupancies.pop(0)
        a_list.pop(0) 
 
    while len(total_occupancies) > 0 and total_occupancies[-1] == 0:
        total_occupancies.pop()
        a_list.pop()  
    # front = [item for item in new_slice_ids[:12] if item in a_list]
    # back = [item for item in new_slice_ids[-12:] if item in a_list]
    # middle = new_slice_ids[12:-12]  
    new_slice_ids = a_list
    return new_slice_ids

def distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def group_atoms_by_column(atoms, threshold=0.01):

    grouped_atoms = {}

    for atom_type, atom_list in atoms.items():
        for atom in atom_list:
            x, y, z, occupancy, Uiso = atom
            found = False

            for key in grouped_atoms.keys():
                if key[0] == atom_type and distance(x, y, key[1], key[2]) < threshold:
                    grouped_atoms[key].append((x, y, z, occupancy, Uiso))
                    found = True
                    break

            if not found:
                key = (atom_type, x, y)
                grouped_atoms[key] = [(x, y, z, occupancy, Uiso)]

    return grouped_atoms

def ungroup_atoms(grouped_atoms):

    atoms = {}

    for key, atom_list in grouped_atoms.items():
        atom_type = key[0]
        if atom_type not in atoms:
            atoms[atom_type] = []
        atoms[atom_type].extend(atom_list)

    return atoms

def calculate_quotients_and_remainders(control_list, counts_list):
    quotients = []
    remainders = []

    # Ensure both lists are of the same length
    if len(control_list) != len(counts_list):
        raise ValueError("control_list and counts_list must be of the same length")

    for control, count in zip(control_list, counts_list):
        quotient = control // count  # Integer division
        remainder = control % count  # Modulo operation

        quotients.append(int(quotient))
        remainders.append(int(remainder))
    return quotients, remainders

def collect_counts(crystal_dict):
    counts = []
    for value in crystal_dict.values():
        counts.append(len(value))
    return counts
def modify_occupancy(all_dicts, quotients, remainders):
    for i, (quotient, remainder) in enumerate(zip(quotients, remainders)):
        keys = list(all_dicts[0].keys())
        key = keys[i]  
        # print(all_dicts,key)
        
        if remainder == 0:
            for j in range(quotient):
                # print(len(all_dicts[j][key]))
                for k in range(len(all_dicts[j][key])):
                    old_tuple = all_dicts[j][key][k]
                    new_tuple = (old_tuple[0], old_tuple[1], old_tuple[2], 1.0, old_tuple[4])
                    all_dicts[j][key][k] = new_tuple

        else:

            for j in range(quotient):
                for k in range(len(all_dicts[j][key])):
                    old_tuple = all_dicts[j][key][k]
                    new_tuple = (old_tuple[0], old_tuple[1], old_tuple[2], 1.0, old_tuple[4])
                    all_dicts[j][key][k] = new_tuple
            
       
            sorted_tuples = sorted(all_dicts[quotient][key], key=lambda x: x[2])
            modified_count = 0

            new_tuple_list = []
            for old_tuple in sorted_tuples:
                if modified_count < remainder:
                    new_tuple = (old_tuple[0], old_tuple[1], old_tuple[2], 1.0, old_tuple[4])
                    new_tuple_list.append(new_tuple)
                    modified_count += 1
                else:
                    new_tuple_list.append(old_tuple)

            all_dicts[quotient][key] = new_tuple_list

def modify_occupancy2(atoms1, atoms2, atoms3, atoms4, atoms5, atoms6, atoms7, new_slice_ids):
    # Step 1: Generate 16 slices from the input atoms
    slice0, slice1, slice2, slice3 = slice_dict_by_z(atoms1)
    slice4, slice5, slice6, slice7 = slice_dict_by_z(atoms2)
    slice8, slice9, slice10, slice11 = slice_dict_by_z(atoms3)
    slice12, slice13, slice14, slice15 = slice_dict_by_z(atoms4)
    slice16, slice17, slice18, slice19 = slice_dict_by_z(atoms5)
    slice20, slice21, slice22, slice23 = slice_dict_by_z(atoms6)
    slice24, slice25, slice26, slice27 = slice_dict_by_z(atoms7)

    # Combine all slices into a list
    slices = [
        slice0, slice1, slice2, slice3, slice4, slice5, slice6, slice7, 
        slice8, slice9, slice10, slice11, slice12, slice13, slice14, slice15,
        slice16, slice17, slice18, slice19, slice20, slice21, slice22, slice23,
        slice24, slice25, slice26, slice27
    ]

    # Reorder slices based on new_slice_ids
    new_slice = [x for x in new_slice_ids]     
    reordered_slices = []
    for i in new_slice:
        reordered_slices.append(slices[i])
        # print(i, slices[i],'\n')
 
    # Helper function to calculate the minimum distance between two layers
    def get_min_distance(layer1, layer2):
        min_distance = float('inf')
        for element1, positions1 in layer1.items():
            for element2, positions2 in layer2.items():
                for (x1, y1, _, occ1, _) in positions1:
                    for (x2, y2, _, occ2, _) in positions2:
                        if occ1 == 1 and occ2 == 1:
                            distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                    
                            min_distance = min(min_distance, distance)
        return min_distance

    def get_nearest_atom(layer1_atom, layer2):
        x1, y1, z1, occ1, db1 = layer1_atom
        min_distance = float('inf')
        nearest_atom = None
        
        for positions in layer2.values():
            for x2, y2, z2, occ2, db2 in positions:
                distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                if distance < min_distance:
                    min_distance = distance
                    nearest_atom = (x2, y2, z2, occ2, db2)
        return nearest_atom

    # Helper function to set all atoms in a layer to occupancy 0
    def set_layer_to_zero(layer):
        for element in layer:
            layer[element] = [(x, y, z, 0.0, db) for (x, y, z, _, db) in layer[element]]

    # Helper function to check if all atoms in a layer have occupancy 1
    def all_atoms_occupied(layer):
        for positions in layer.values():
            if any(occ != 1 for (_, _, _, occ, _) in positions):
                return False
        return True

    # Helper function to check if all atoms in a layer have occupancy 0
    def all_atoms_unoccupied(layer):
        for positions in layer.values():
            if any(occ != 0 for (_, _, _, occ, _) in positions):
                return False
        return True

    # Step 1: Iterate from front to back (inner to outer)
    # while True:
    #     changes_made = False  # Track if any changes are made in this iteration
    i = 0
    while i < len(reordered_slices)/2 - 1:
        layer1 = reordered_slices[i]
        layer2 = reordered_slices[i + 1]
        layer3 = reordered_slices[i + 2]
    
        # Skip layers with all occupancy 0
        if all_atoms_unoccupied(layer1):
            i += 1
            continue
      
        for element1, positions1 in layer1.items():
            for idx, atom1 in enumerate(positions1):
                x1, y1, z1, occ1, db1 = atom1
                # Only check atoms with occupancy 1
                if occ1 == 1:
                    nearest_atom = get_nearest_atom(atom1, layer2)
                    if nearest_atom is not None:
                        _, _, _, occ2, _ = nearest_atom
                        # If nearest atom in layer2 has occupancy 0, set current atom's occupancy to 0
                        if occ2 == 0:
                            positions1[idx] = (x1, y1, z1, 0.0, db1)         
                            
        for element1, positions1 in layer1.items():
            for idx, atom1 in enumerate(positions1):
                x1, y1, z1, occ1, db1 = atom1
                # Only check atoms with occupancy 1
                if occ1 == 1:
                    nearest_atom_3 = get_nearest_atom(atom1, layer3)
                    if nearest_atom_3 is not None:
                        _, _, _, occ3, _ = nearest_atom_3
                   
                        if occ3 == 0:
                            positions1[idx] = (x1, y1, z1, 0.0, db1)
 
        if i > 0:  
            layer_prev = reordered_slices[i - 1]
 
            for element1, positions1 in layer1.items():
                for idx, atom1 in enumerate(positions1):
                    x1, y1, z1, occ1, db1 = atom1
    
                    if occ1 == 0:  
                        nearest_prev_atom = get_nearest_atom(atom1, layer_prev)  
    
                        if nearest_prev_atom is not None:
                            x_p, y_p, z_p, occ_p, db_p = nearest_prev_atom
    
                            if occ_p == 1:  
                               
                                for element_p, positions_p in layer_prev.items():
                                    for k, atom_prev in enumerate(positions_p):
                                        if atom_prev[:3] == (x_p, y_p,z_p): 
                                            layer_prev[element_p][k] = (x_p, y_p, z_p, 0.0, db_p) 
                                            break
        
        i +=1
    i = 0
    while i < len(reordered_slices)/2 - 1:
        layer1 = reordered_slices[i]
        layer2 = reordered_slices[i + 1]                        
        min_distance = get_min_distance(layer1, layer2)
        # print('min',min_distance)
        if all_atoms_unoccupied(layer1):         
            i += 1         
            continue
        
        if min_distance > 0.6:
            # If current layer does not have all atoms occupied, reset previous layers
            if not all_atoms_occupied(layer1):
                for j in range(0, i + 1):
                    set_layer_to_zero(reordered_slices[j])
                # Reset the iteration to start from the beginning
                i = 0
            else:
                # Current layer has all atoms occupied, stop further checks
                break
        else:
            i += 1
    
    # Step 2: Iterate from back to front (outer to inner)
    i = len(reordered_slices) - 1
    while i > (len(reordered_slices)/2 - 1) :
        layer1 = reordered_slices[i]
        layer2 = reordered_slices[i - 1]
        layer3 = reordered_slices[i - 2]
    
        # Skip layers with all occupancy 0
        if all_atoms_unoccupied(layer1):
            i -= 1
            continue
        # Check each atom in the current layer
        
        for element1, positions1 in layer1.items():
            for idx, atom1 in enumerate(positions1):
                x1, y1, z1, occ1, db1 = atom1
                # Only check atoms with occupancy 1
                if occ1 == 1:
                    nearest_atom = get_nearest_atom(atom1, layer2)
                 
                    if nearest_atom is not None:
                        _, _, _, occ2, _ = nearest_atom
    
                        # If nearest atom in layer2 has occupancy 0, set current atom's occupancy to 0
                        if occ2 == 0:
                            positions1[idx] = (x1, y1, z1, 0.0, db1)
                            
        for element1, positions1 in layer1.items():
            for idx, atom1 in enumerate(positions1):
                x1, y1, z1, occ1, db1 = atom1
         
                if occ1 == 1:
                    nearest_atom_3 = get_nearest_atom(atom1, layer3)
    
                    if nearest_atom_3 is not None:
                        _, _, _, occ3, _ = nearest_atom_3
                   
                        if occ3 == 0:
                            positions1[idx] = (x1, y1, z1, 0.0, db1)
                            
        if i < len(reordered_slices)-1: 
            layer_prev = reordered_slices[i + 1]
            
            for element1, positions1 in layer1.items():
                for idx, atom1 in enumerate(positions1):
                    x1, y1, z1, occ1, db1 = atom1
    
                    if occ1 == 0:  
                        nearest_prev_atom = get_nearest_atom(atom1, layer_prev)  
    
                        if nearest_prev_atom is not None:
                            x_p, y_p, z_p, occ_p, db_p = nearest_prev_atom
    
                            if occ_p == 1:  
                                for element_p, positions_p in layer_prev.items():
                                    for k, atom_prev in enumerate(positions_p):
                                        if atom_prev[:3] == (x_p, y_p,z_p):  
                                            layer_prev[element_p][k] = (x_p, y_p, z_p, 0.0, db_p) 
                                            break
   
        i -=1
    i = len(reordered_slices) - 1 
    while i > (len(reordered_slices)/2 - 1) :
        layer1 = reordered_slices[i]
        layer2 = reordered_slices[i - 1]
        min_distance = get_min_distance(layer1, layer2)
        if all_atoms_unoccupied(layer1):         
            i -= 1         
            continue
           
        if min_distance > 0.6:
            # If current layer does not have all atoms occupied, reset later layers
            if not all_atoms_occupied(layer1):
                for j in range(i, len(reordered_slices)):
                    set_layer_to_zero(reordered_slices[j])
                # Reset the iteration to start from the end
                i = len(reordered_slices) - 1
            else:
                # Current layer has all atoms occupied, stop further checks
                break
        else:
            i -= 1
        # if not changes_made:
        #         break
    restored_slices = reordered_slices
    
    modified_atoms1 = recover_dict_from_slices(*restored_slices[:4])
    modified_atoms2 = recover_dict_from_slices(*restored_slices[4:8])
    modified_atoms3 = recover_dict_from_slices(*restored_slices[8:12])
    modified_atoms4 = recover_dict_from_slices(*restored_slices[12:16])
    modified_atoms5 = recover_dict_from_slices(*restored_slices[16:20])
    modified_atoms6 = recover_dict_from_slices(*restored_slices[20:24])
    modified_atoms7 = recover_dict_from_slices(*restored_slices[24:])
  
    return modified_atoms1, modified_atoms2, modified_atoms3, modified_atoms4, modified_atoms5, modified_atoms6, modified_atoms7

def loss_selc(control_list,atoms,C1, abf):
    loss = []

    new_structure = [24,25,22,23,16,17,14,15,8,9,6,7,0,1,2,3,4,5,10,11,12,13,18,19,20,21,26,27]
   
    Loss = simulation(control_list,atoms,new_structure,C1, abf)
    loss.append(Loss)
    loss_tensor = torch.tensor(loss)
    return torch.max(loss_tensor)
def simulation(control_list,atoms,new_slice_ids,C1, abf):
    slice_name_00 = root_path+folder+'/column6/'+'t01/task/slc_00/BTO'
    slice_folder_00 = root_path+folder+'/column6/'+'t01/task/slc_00/'
    slice_name = root_path+folder+'/column6/'+'t01/task/slc/BTO'
    slice_folder = root_path+folder+'/column6/'+'t01/task/slc/'
    
    slice_name_01 = root_path+folder+'/column6/'+'t01/task/slc_01/BTO'
    slice_folder_01 = root_path+folder+'/column6/'+'t01/task/slc_01/'
    slice_name_02 = root_path+folder+'/column6/'+'t01/task/slc_02/BTO'
    slice_folder_02 = root_path+folder+'/column6/'+'t01/task/slc_02/'
    slice_name_03 = root_path+folder+'/column6/'+'t01/task/slc_03/BTO'
    slice_folder_03 = root_path+folder+'/column6/'+'t01/task/slc_03/'
    slice_name_04 = root_path+folder+'/column6/'+'t01/task/slc_04/BTO'
    slice_folder_04 = root_path+folder+'/column6/'+'t01/task/slc_04/'
    slice_name_05 = root_path+folder+'/column6/'+'t01/task/slc_05/BTO'
    slice_folder_05 = root_path+folder+'/column6/'+'t01/task/slc_05/'
    slice_name_06 = root_path+folder+'/column6/'+'t01/task/slc_06/BTO'
    slice_folder_06 = root_path+folder+'/column6/'+'t01/task/slc_06/'
    msa_prm = root_path+folder+'/column6/'+ 't01/task/msa.prm'

    output_path= root_path+folder+'/column6/'+'t01/'
    if not os.path.isdir(output_path):
        os.makedirs(output_path, exist_ok=True)
    
    output_file_01 = output_path+'BTO_new_01.cel'
    output_file_02 = output_path+'BTO_new_02.cel'
    output_file_03 = output_path+'BTO_new_03.cel'
    output_file_04 = output_path+'BTO_new_04.cel'
    output_file_05 = output_path+'BTO_new_05.cel'
    output_file_06 = output_path+'BTO_new_06.cel'
    output_file_07 = output_path+'BTO_new_07.cel'
    
    grouped_atoms = group_atoms_by_column(atoms, threshold=0.1)
            
    a1 = copy.deepcopy(grouped_atoms)
    a2 = copy.deepcopy(grouped_atoms)
    a3 = copy.deepcopy(grouped_atoms)
    a4 = copy.deepcopy(grouped_atoms)
    a5 = copy.deepcopy(grouped_atoms)
    a6 = copy.deepcopy(grouped_atoms)
    a7 = copy.deepcopy(grouped_atoms)
    # List of dictionaries
    all_dicts = [a1, a2, a3, a4, a5, a6, a7]
    counts_list = collect_counts(grouped_atoms)
    print('counts_list:',counts_list)
    quotients, remainders = calculate_quotients_and_remainders(control_list, counts_list)
    print(quotients, remainders)
    modify_occupancy(all_dicts, quotients, remainders)

    atoms_01 = ungroup_atoms(a1)
    atoms_02 = ungroup_atoms(a2)
    atoms_03 = ungroup_atoms(a3)
    atoms_04 = ungroup_atoms(a4)
    atoms_05 = ungroup_atoms(a5)
    atoms_06 = ungroup_atoms(a6)
    atoms_07 = ungroup_atoms(a7)
    
    # BuildCell(output_file = output_file_01,lattice=lattice, atoms = atoms_01, override=True, output = True)        
    # BuildCell(output_file = output_file_02,lattice=lattice, atoms = atoms_02, override=True, output = True)  
    # BuildCell(output_file = output_file_03,lattice=lattice, atoms = atoms_03, override=True, output = True)  
    # BuildCell(output_file = output_file_04,lattice=lattice, atoms = atoms_04, override=True, output = True)  
    # BuildCell(output_file = output_file_05,lattice=lattice, atoms = atoms_05, override=True, output = True)  
    # BuildCell(output_file = output_file_06,lattice=lattice, atoms = atoms_06, override=True, output = True) 
    # BuildCell(output_file = output_file_07,lattice=lattice, atoms = atoms_07, override=True, output = True) 
    
    # slice_ids = new_slice_ids[:12] + new_slice_ids[-12:]
    slice_ids = new_slice_ids
    at1, at2, at3, at4, at5, at6, at7 = modify_occupancy2(atoms_01, atoms_02, atoms_03, atoms_04, atoms_05, atoms_06, atoms_07, slice_ids)

    atoms_list = [at1, at2, at3, at4, at5, at6, at7]
    print('wu:',new_slice_ids)
    new_slice_ids = filter_layers(atoms_list, new_slice_ids)
    print('new',new_slice_ids)

    BuildCell(output_file = output_file_01,lattice=lattice, atoms = at1, override=True, output = True)        
    BuildCell(output_file = output_file_02,lattice=lattice, atoms = at2, override=True, output = True)  
    BuildCell(output_file = output_file_03,lattice=lattice, atoms = at3, override=True, output = True)  
    BuildCell(output_file = output_file_04,lattice=lattice, atoms = at4, override=True, output = True)  
    BuildCell(output_file = output_file_05,lattice=lattice, atoms = at5, override=True, output = True)  
    BuildCell(output_file = output_file_06,lattice=lattice, atoms = at6, override=True, output = True) 
    BuildCell(output_file = output_file_07,lattice=lattice, atoms = at7, override=True, output = True)
    
    abf = abf
    tilt_x = 0.30442937963538874
    tilt_y = 0.015341411653023385
    vibration = 0.0155
    vibration_matrix = (1,vibration,vibration,0)
    thick  = len(new_slice_ids)              
    ht = 300
    wlkev = 1.2398419843320025  # c * h / q_e * 1E6
    twomc2 = 1021.9978999923284  # 2*m0*c**2 / q_e
    
    STF_HT2WL = wlkev / math.sqrt(ht * (twomc2 + ht))

    aberrations_dict = {1: (C1,0.0000), 2:(-1.1539699519764115,2.293596885253435), 3:(-75.33994898659485,64.95604867495541), 
                          4:(62.26177026746993, -33.04935804364307),5:(-5000,0),6:(1999.9577,-2000.0000),7:(-1999.1128,-1998.7757), 11:(2000000,0)}

    wave_output = root_path+folder+'/column6/'+'t01/task/wav/BTO.wav'
    wavimg_prm = root_path+folder+'/column6/'+'t01/task/wavimg.prm'
    
    drp.commands.celslc(cel_file=output_file_01, # location of cel file
                slice_name=slice_name_00, # target file names
                nx=nx,        # number of sampling points along x
                ny=ny,        # number of sampling points along y
                nz=nz,         # number of sampling points along z
                ht=300,       # high tension
                abf=abf,
                absorb=False,  # apply absorptive form factors
                dwf=True,     # apply Debye-Waller factors
                output=True,  # Command line output (prints executed command)
                pot=True      # Saves potentials
               )
    if not os.path.exists(slice_name):
            os.makedirs(slice_name)

    file_names00 = {
        'BTO_001.sli': 'BTO_025.sli',
        'BTO_002.sli': 'BTO_026.sli',
        'BTO_003.sli': 'BTO_023.sli',
        'BTO_004.sli': 'BTO_024.sli',
    }

    # Rename each file
    for old_name, new_name in file_names00.items():
        old_path = os.path.join(slice_folder_00, old_name)
        new_path = os.path.join(slice_folder_00, new_name)
        if os.path.exists(old_path):
            os.rename(old_path, new_path)
            # print(f'File "{old_name}" renamed to "{new_name}".')
        else:
            print(f'File "{old_name}" not found in directory.')

    files_to_copy_00 = ['BTO_025.sli', 'BTO_026.sli', 'BTO_023.sli','BTO_024.sli']

    for file_name in files_to_copy_00:
        source_file = os.path.join(slice_folder_00, file_name)
        if os.path.exists(source_file):
            shutil.copy(source_file, slice_folder)
            print(f'Copied {file_name} to {slice_folder}')
        else:
            print(f'File {file_name} not found in source directory')
    
    
    drp.commands.celslc(cel_file=output_file_02, # location of cel file
            slice_name=slice_name_01, # target file names
            nx=nx,        # number of sampling points along x
            ny=ny,        # number of sampling points along y
            nz=nz,         # number of sampling points along z
            ht=ht,       # high tension
            abf=abf,
            absorb=False,  # apply absorptive form factors
            dwf=True,     # apply Debye-Waller factors
            output=True,  # Command line output (prints executed command)
            pot=True      # Saves potentials
           )

    drp.commands.celslc(cel_file=output_file_03, # location of cel file
            slice_name=slice_name_02, # target file names
            nx=nx,        # number of sampling points along x
            ny=ny,        # number of sampling points along y
            nz=nz,         # number of sampling points along z
            ht=ht,       # high tension
            abf=abf,
            absorb=False,  # apply absorptive form factors
            dwf=True,     # apply Debye-Waller factors
            output=True,  # Command line output (prints executed command)
            pot=True      # Saves potentials
           )

    drp.commands.celslc(cel_file=output_file_04, # location of cel file
            slice_name=slice_name_03, # target file names
            nx=nx,        # number of sampling points along x
            ny=ny,        # number of sampling points along y
            nz=nz,         # number of sampling points along z
            ht=ht,       # high tension
            abf=abf,
            absorb=False,  # apply absorptive form factors
            dwf=True,     # apply Debye-Waller factors
            output=True,  # Command line output (prints executed command)
            pot=True      # Saves potentials
           )
    drp.commands.celslc(cel_file=output_file_05, # location of cel file
            slice_name=slice_name_04, # target file names
            nx=nx,        # number of sampling points along x
            ny=ny,        # number of sampling points along y
            nz=nz,         # number of sampling points along z
            ht=ht,       # high tension
            abf=abf,
            absorb=False,  # apply absorptive form factors
            dwf=True,     # apply Debye-Waller factors
            output=True,  # Command line output (prints executed command)
            pot=True      # Saves potentials
           )
    
    drp.commands.celslc(cel_file=output_file_06, # location of cel file
            slice_name=slice_name_05, # target file names
            nx=nx,        # number of sampling points along x
            ny=ny,        # number of sampling points along y
            nz=nz,         # number of sampling points along z
            ht=ht,       # high tension
            abf=abf,
            absorb=False,  # apply absorptive form factors
            dwf=True,     # apply Debye-Waller factors
            output=True,  # Command line output (prints executed command)
            pot=True      # Saves potentials
           )
    drp.commands.celslc(cel_file=output_file_07, # location of cel file
            slice_name=slice_name_06, # target file names
            nx=nx,        # number of sampling points along x
            ny=ny,        # number of sampling points along y
            nz=nz,         # number of sampling points along z
            ht=ht,       # high tension
            abf=abf,
            absorb=False,  # apply absorptive form factors
            dwf=True,     # apply Debye-Waller factors
            output=True,  # Command line output (prints executed command)
            pot=True      # Saves potentials
           )
              
          
    file_names = {
        'BTO_001.sli': 'BTO_017.sli',
        'BTO_002.sli': 'BTO_018.sli',
        'BTO_003.sli': 'BTO_015.sli',
        'BTO_004.sli': 'BTO_016.sli',
    }

    # Rename each file
    for old_name, new_name in file_names.items():
        old_path = os.path.join(slice_folder_01, old_name)
        new_path = os.path.join(slice_folder_01, new_name)
        if os.path.exists(old_path):
            os.rename(old_path, new_path)
            # print(f'File "{old_name}" renamed to "{new_name}".')
        else:
            print(f'File "{old_name}" not found in directory.')
 
    file_names_01 = {
        'BTO_001.sli': 'BTO_009.sli',
        'BTO_002.sli': 'BTO_010.sli',
        'BTO_003.sli': 'BTO_007.sli',
        'BTO_004.sli': 'BTO_008.sli',
    }

    # Rename each file
    for old_name, new_name in file_names_01.items():
        old_path = os.path.join(slice_folder_02, old_name)
        new_path = os.path.join(slice_folder_02, new_name)
        if os.path.exists(old_path):
            os.rename(old_path, new_path)
            # print(f'File "{old_name}" renamed to "{new_name}".')
        else:
            print(f'File "{old_name}" not found in directory.')
    file_names_01 = {
        'BTO_001.sli': 'BTO_001.sli',
        'BTO_002.sli': 'BTO_002.sli',
        'BTO_003.sli': 'BTO_003.sli',
        'BTO_004.sli': 'BTO_004.sli',
    }

    # Rename each file
    for old_name, new_name in file_names_01.items():
        old_path = os.path.join(slice_folder_03, old_name)
        new_path = os.path.join(slice_folder_03, new_name)
        if os.path.exists(old_path):
            os.rename(old_path, new_path)
            # print(f'File "{old_name}" renamed to "{new_name}".')
        else:
            print(f'File "{old_name}" not found in directory.')
          
    file_names_01 = {
        'BTO_001.sli': 'BTO_005.sli',
        'BTO_002.sli': 'BTO_006.sli',
        'BTO_003.sli': 'BTO_011.sli',
        'BTO_004.sli': 'BTO_012.sli',
    }

    # Rename each file
    for old_name, new_name in file_names_01.items():
        old_path = os.path.join(slice_folder_04, old_name)
        new_path = os.path.join(slice_folder_04, new_name)
        if os.path.exists(old_path):
            os.rename(old_path, new_path)
            # print(f'File "{old_name}" renamed to "{new_name}".')
        else:
            print(f'File "{old_name}" not found in directory.')
     
    file_names_01 = {
        'BTO_001.sli': 'BTO_013.sli',
        'BTO_002.sli': 'BTO_014.sli',
        'BTO_003.sli': 'BTO_019.sli',
        'BTO_004.sli': 'BTO_020.sli',
    }

    # Rename each file
    for old_name, new_name in file_names_01.items():
        old_path = os.path.join(slice_folder_05, old_name)
        new_path = os.path.join(slice_folder_05, new_name)
        if os.path.exists(old_path):
            os.rename(old_path, new_path)
            # print(f'File "{old_name}" renamed to "{new_name}".')
        else:
            print(f'File "{old_name}" not found in directory.')   
    file_names_01 = {
        'BTO_001.sli': 'BTO_021.sli',
        'BTO_002.sli': 'BTO_022.sli',
        'BTO_003.sli': 'BTO_027.sli',
        'BTO_004.sli': 'BTO_028.sli',
    }

    # Rename each file
    for old_name, new_name in file_names_01.items():
        old_path = os.path.join(slice_folder_06, old_name)
        new_path = os.path.join(slice_folder_06, new_name)
        if os.path.exists(old_path):
            os.rename(old_path, new_path)
            # print(f'File "{old_name}" renamed to "{new_name}".')
        else:
            print(f'File "{old_name}" not found in directory.')
            

    files_to_copy_01 = ['BTO_017.sli', 'BTO_018.sli', 'BTO_015.sli','BTO_016.sli']
    for file_name in files_to_copy_01:
        source_file = os.path.join(slice_folder_01, file_name)
        if os.path.exists(source_file):
            shutil.copy(source_file, slice_folder)
            # print(f'Copied {file_name} to {slice_folder}')
        else:
            print(f'File {file_name} not found in source directory')

    files_to_copy_02 = ['BTO_009.sli','BTO_010.sli','BTO_007.sli','BTO_008.sli']
    for file_name in files_to_copy_02:
        source_file = os.path.join(slice_folder_02, file_name)
        if os.path.exists(source_file):
            shutil.copy(source_file, slice_folder)
            # print(f'Copied {file_name} to {slice_folder}')
        else:
            print(f'File {file_name} not found in source directory')

    files_to_copy_03 = ['BTO_001.sli','BTO_002.sli','BTO_003.sli','BTO_004.sli']
    for file_name in files_to_copy_03:
        source_file = os.path.join(slice_folder_03, file_name)
        if os.path.exists(source_file):
            shutil.copy(source_file, slice_folder)
        else:
            print(f'File {file_name} not found in source directory')
    
    files_to_copy_04 = ['BTO_005.sli','BTO_006.sli','BTO_011.sli','BTO_012.sli']
    for file_name in files_to_copy_04:
        source_file = os.path.join(slice_folder_04, file_name)
        if os.path.exists(source_file):
            shutil.copy(source_file, slice_folder)
        else:
            print(f'File {file_name} not found in source directory')
    
    files_to_copy_05 = ['BTO_013.sli','BTO_014.sli','BTO_019.sli','BTO_020.sli']
    for file_name in files_to_copy_05:
        source_file = os.path.join(slice_folder_05, file_name)
        if os.path.exists(source_file):
            shutil.copy(source_file, slice_folder)
        else:
            print(f'File {file_name} not found in source directory')
    
    files_to_copy_06 = ['BTO_021.sli','BTO_022.sli','BTO_027.sli','BTO_028.sli']
    for file_name in files_to_copy_06:
        source_file = os.path.join(slice_folder_06, file_name)
        if os.path.exists(source_file):
            shutil.copy(source_file, slice_folder)
        else:
            print(f'File {file_name} not found in source directory')
    
            
    msa = drp.msaprm.MsaPrm()
    msa.conv_semi_angle = 0.2
    msa.wavelength = STF_HT2WL  # wavelength (in nm) for 300 keV electrons
    msa.h_scan_frame_size = a 
    msa.v_scan_frame_size = b 
    msa.scan_columns = nx
    msa.scan_rows = ny
    msa.tilt_x = tilt_x # object tilt along x in degree
    msa.tilt_y = tilt_y # object tilt along y in degree
    msa.focus_spread=3
    msa.slice_files = slice_name # location of phase gratings
    msa.number_of_slices = 28 # Number of slices in phase gratings
    msa.det_readout_period = 1 # Readout after every 2 slices
    msa.tot_number_of_slices = thick # Corresponds to 5 layers / unit cells of SrTiO3
    msa.save_msa_prm(msa_prm)
    prm_filename = msa_prm
    msa_parameters = load_mas(prm_filename)     
    msa_parameters['slice_ids'] = new_slice_ids
    # msa_parameters['conv_semi_angle'] = 20
    save_msa_prm(prm_filename,msa_parameters)

    drp.commands.msa(prm_file = msa_prm, 
         output_file= wave_output,
         ctem=True, # Flag for conventional TEM simulation (otherwise calculates STEM images)
         output=True)
    wavimg = drp.WavimgPrm()
    sl = round(thick) # Perform simulation for slice # 2
    print(sl)

    wave_files = root_path+folder+'/'+'column6/t01/task/wav/BTO_sl{:03d}.wav'.format(sl)
    output_files = root_path+folder+'/'+'column6/t01/task/img/BTO_sl{:03d}.dat'.format(sl)
    stl = '{:03d}'.format(sl)
    fake_data = root_path+folder+'/'+'column6/t01/task/img/'+'BTO_sl'+stl+'.dat'
    # Setup (fundamental) MSA parameters
    wavimg.high_tension = ht
    wavimg.mtf = (1, 1.068, 'MTF-US2k-300.mtf')  
    wavimg.oa_radius = 250 # Apply objective aperture [mrad]
    wavimg.oa_position = (0, 0)
    # wavimg.vibration = vibration_matrix # Apply isotropic image spread of 16 pm rms displacement
    wavimg.vibration = vibration_matrix
    wavimg.spat_coherence = (1, 0.4627934051941923)
    wavimg.temp_coherence = (1, 4) # Apply spatial coherence
    wavimg.wave_sampling = (a / nx, b / ny)
    wavimg.wave_files = wave_files 
    # wavimg.img_frame_offset= (8.7, -0.5)

    wavimg.wave_dim = (nx, ny)
    wavimg.aberrations_dict = aberrations_dict # apply aberrations
    wavimg.output_files = output_files # Simulate for slice number "sl"
    wavimg.output_format = 0
    wavimg.flag_spec_frame = 1
    wavimg.output_dim = (64, 45)             
    wavimg.output_sampling = (0.00887, 0.00887)

    # Save wavimg Parameter file
    wavimg.save_wavimg_prm(wavimg_prm)
    drp.commands.wavimg(wavimg_prm, output=True)

    img_fake = np.fromfile(fake_data, dtype='float32').reshape((45, 64))
    fkl = img_fake
    y_fake_p = torch.tensor(fkl, device = device, dtype = dtype)
    Loss = -F.mse_loss(y_real_p,y_fake_p)
    display.clear_output(wait=True)
    return Loss
class DrProbe(SyntheticTestFunction):
    _optimal_value = 0.0
    _check_grad_at_opt: bool = False
    _optimizers: List[Tuple[float, ...]]

    def __init__(self, dim: int = 3, noise_std: Optional[float] = None, negate: bool = False) -> None:
        self.dim = dim
        self._bounds = [(-0.5, 0.5) for _ in range(self.dim)]
        self._optimizers = [tuple(0.0 for _ in range(self.dim))]
        super().__init__(noise_std=noise_std, negate=negate)
    
    def evaluate_true(self, X, noise=0.1):
        loss = []
        # cel_file = io.read_input_file(folder)
        
        if X.ndim <= 1:
            X = X.unsqueeze(0)
        for folder_num, x in enumerate(X):
            Num_x = [i.item() for i in x]
            print(Num_x)
            l1 = Num_x[0]
            l2 = Num_x[1]
            l3 = Num_x[2]
            l4 = Num_x[3]
            l5 = Num_x[4]
            l6 = Num_x[5]
            l7 = Num_x[6]
            l8 = Num_x[7]
            
            C1 = Num_x[8]
            abf = Num_x[9]
           
            control_list = [l1, l2, l3, l4, l5, l6, l7, l8]
            apo = np.genfromtxt('STO01/position/t101/BTO_n.cel', dtype=str, skip_header=2, skip_footer=1, usecols=(0, 1, 2, 3, 4, 5))
            atoms = {}
            for item in apo:
                symbol = item[0].strip()  # Remove any leading or trailing whitespace
                coords = tuple(map(float, item[1:5]))  # Convert coordinates to float
                charge = float(item[5])  # Convert charge to float
                if symbol not in atoms:
                    atoms[symbol] = []
                atoms[symbol].append(coords + (charge,))
            updated_atoms = {
                                symbol: [(x, y, z, 0, vibration * 100) for x, y, z, w, vibration in coords_list]
                                for symbol, coords_list in atoms.items()
                            }
            print(control_list)
            Loss = loss_selc(control_list,updated_atoms,C1,abf)
            loss.append(Loss) 
        display.clear_output(wait=True)
        return torch.tensor(loss, dtype=dtype, device=device)

def eval_objective_DrProb(x):
    """This is a helper function we use to unnormalize and evalaute a point"""   
    return fun_q(x)

fun_q = DrProbe(dim=10, negate=False).to(dtype=dtype, device=device) 
fun_q.bounds[0, :] = torch.tensor([1,1,1,1,1,1,1,1,  3.5, 0.01]).to(dtype=dtype, device=device) 
fun_q.bounds[1, :] = torch.tensor([7,7,14,7,14,7,7,7,5.5, 0.05]).to(dtype=dtype, device=device) 
print(fun_q.bounds) 
dim_q = fun_q.dim
lb, ub = fun_q.bounds
n_init = 4 * dim_q

@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 1.6
    length_min: float = 0.1**5
    length_max: float = 100
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 10  # Note: The original paper uses 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )
        
def update_state(state, Y_next):
    if max(Y_next) > state.best_value + 1e-2 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    state.best_value = max(state.best_value, max(Y_next).item())
    if state.length < state.length_min:
        state.restart_triggered = True
    return state

state_q = TurboState(dim = dim_q, batch_size = 4)
print(state_q)

def get_initial_points(dim, n_pts):
    #sobol = SobolEngine(dimension=dim, scramble=True, seed=0)
    sobol = SobolEngine(dimension=dim, scramble=True)
    X_init = sobol.draw(n=n_pts).to(dtype=dtype, device=device)
    return X_init

def restarts_BO_q(root_path=root_path):

    with open(root_path+folder+'/'+'column6/t01/'+'new_resulst.txt') as f:
        candi_lines = f.readlines()

    _X = []
    _Y = []
    for i in candi_lines:
        info = i.split()
        file_name = info[0]
        _x = np.array([float(num) for num in info[1].split('_') if num])
        _y = float(info[2])
                        
        _X.append(_x)
        _Y.append(_y)
        #candi_dict[file_name] = [x_term, y]

    tensor_x = normalize(torch.tensor(_X, dtype=dtype, device=device), fun_q.bounds)
    tensor_y = torch.tensor(_Y, dtype=dtype, device=device).unsqueeze(-1)

    return {'x':tensor_x, 'y':tensor_y}

def generate_batch(
    state,
    model,  # GP model
    X,  # Evaluated points on the domain [0, 1]^d
    Y,  # Function values
    batch_size,
    n_candidates=None,  # Number of candidates for Thompson sampling
    num_restarts=2,
    raw_samples=512,
    acqf="qucb",
    hpar=0.2
):
    assert acqf in ("ts", "ei", "ucb", "qucb")
    assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))
    if n_candidates is None:
        n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

    # Scale the TR to be proportional to the lengthscales
    x_center = X[Y.argmax(), :].clone()
    weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
    weights = weights / weights.mean()
    if weights.ndim < 1:
        weights = weights.unsqueeze(-1)
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
    tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)
    
    if acqf == "ts":
        dim = X.shape[-1]
        sobol = SobolEngine(dim, scramble=True)
        pert = sobol.draw(n_candidates).to(dtype=dtype, device=device)
        pert = tr_lb + (tr_ub - tr_lb) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / dim, 1.0)
        mask = torch.rand(n_candidates, dim, dtype=dtype, device=device) <= prob_perturb
        
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1

        # Create candidate points from the perturbations and the mask
        X_cand = x_center.expand(n_candidates, dim).clone()
        X_cand[mask] = pert[mask]

        # Sample on the candidate points
        thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
        X_next = thompson_sampling(X_cand, num_samples=batch_size)


    elif acqf == "ei":
        ei = qExpectedImprovement(model, train_Y.max())
        X_next, acq_value = optimize_acqf(
            ei,
            bounds=torch.stack([tr_lb, tr_ub]),
            q=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )

    elif acqf == "qucb":
        qucb = qUpperConfidenceBound(model, hpar)
        X_next, acq_value = optimize_acqf(
            qucb,
            bounds=torch.stack([tr_lb, tr_ub]),
            q=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )
        
    elif acqf == "ucb":
        ucb = UpperConfidenceBound(model, beta = hpar, record = record)
        
        X_next, acq_value = optimize_acqf_discrete(
            ucb,
            bounds=torch.stack([tr_lb, tr_ub]),
            q=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )

    return {'X_next': X_next, 'acq':acq_value}

def BO_search_DrProb(acqf = 'qucb', length_min=0.1**5, hpar=500):
    batch_size = 8
    n_init = 64
    WARMUP_STEPS = 512 if not SMOKE_TEST else 32
    NUM_SAMPLES = 256 if not SMOKE_TEST else 16
    THINNING = 32
    N_INIT = 4*dim_q
    N_ITERATIONS = 100 if not SMOKE_TEST else 1
    BATCH_SIZE = 5 if not SMOKE_TEST else 1
    
    start_time = time.time()
    process = psutil.Process()
    
    # Get initial RAM usage
    initial_ram = process.memory_info().rss / (1024 * 1024)  # in MB
    ram_usage = [initial_ram]
    if restart is False:        
        X_turbo_q = get_initial_points(dim_q, n_init)
        X_turbo_q = unnormalize(X_turbo_q, fun_q.bounds)
        t_q = X_turbo_q[:,:-2]
        mq = X_turbo_q[:,-2:]
        k_q = torch.floor(t_q)
        theta_q = t_q - k_q
        rq = sample_z_from_theta(theta = theta_q,num_samples=1)
        X_x = rq + k_q
        X_qx = torch.cat((X_x, mq), dim=1)
        Y_turbo_q = eval_objective_DrProb(X_qx).unsqueeze(-1)
        X_q = normalize(X_qx, fun_q.bounds)
       
        
    else:
        turbo_info = restarts_BO_q(root_path)
        X_turbo_q_0 = turbo_info['x']
        X_turbo_q_1 = get_initial_points(dim_q, n_init)
        X_turbo_q = torch.cat((X_turbo_q_0, X_turbo_q_1), dim=0)
        X_turbo_q = unnormalize(X_turbo_q, fun_q.bounds)       
        t_qq = X_turbo_q[:,:-2]
        mqq = X_turbo_q[:,-2:]
        kqq = torch.floor(t_qq)
        thetaqq = t_qq - kqq
        riqq = sample_z_from_theta(theta = thetaqq,num_samples=1)
        X_xx = riqq + kqq
        X_qxx = torch.cat((X_xx, mqq), dim=1)
        Y_turbo_q = eval_objective_DrProb(X_qxx).unsqueeze(-1)
        X_q = normalize(X_qxx, fun_q.bounds)
        
    state = TurboState(dim_q, batch_size=batch_size)

    NUM_RESTARTS = 10 if not SMOKE_TEST else 2
    RAW_SAMPLES = 16384 if not SMOKE_TEST else 4
    N_CANDIDATES = min(5000, max(2000, 200 * dim_q)) if not SMOKE_TEST else 4
    parameter_results = {}
    torch.manual_seed(0)
    iteration_times = [] 
    best_values = []   
    
    print(X_q.shape)

    try:
        while not state.restart_triggered:  # Run until TuRBO converges
            # Fit a GP model
            train_Y_q = (Y_turbo_q - Y_turbo_q.mean()) / Y_turbo_q.std()
                    
            likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
            
            # covar_module = ScaleKernel(SpectralDeltaKernel(num_dims=dim_q, num_deltas=32))
            covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
                MaternKernel(
                    nu=2.5, ard_num_dims=dim_q, lengthscale_constraint=Interval(0.005, 4.0)
                )
            )
           
            train_Yvar_q = torch.full_like(train_Y_q, 0.05)
            model = FixedNoiseGP(X_q, train_Y_q, train_Yvar_q, covar_module=covar_module)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            # cat_dims = [0, 1, 2, 3]
            # model = MixedSingleTaskGP(X_turbo_q, train_Y_q, cat_dims=cat_dims,train_Yvar=train_Yvar_q)
            
            # model = FixedNoiseGP(X_turbo_q, train_Y_q, train_Yvar_q, covar_module=covar_module)
            # mll = ExactMarginalLogLikelihood(model.likelihood, model)

            fit_gpytorch_model(mll)
            
            acqf = 'qucb'
            print(X_q)

            # Create a batch
            X_info = generate_batch(
                state=state,
                model=model,
                X=X_q,
                Y=train_Y_q,
                batch_size=batch_size,
                n_candidates=N_CANDIDATES,
                num_restarts=NUM_RESTARTS,
                raw_samples=RAW_SAMPLES,
                acqf=acqf,
                hpar=hpar,
                )
            X_next = X_info['X_next']
            
            t_next = unnormalize(X_next, fun_q.bounds)
            t_nq = t_next[:,:-2]
            re = t_next[:,-2:]
            k_next = torch.floor(t_nq)
            theta_next = t_nq - k_next           
            ri_next = sample_z_from_theta(theta = theta_next,num_samples=1)
            X_kk = ri_next + k_next
            # print(X_nn)
            X_mm= torch.cat((X_kk, re), dim=1)

            Y_next = eval_objective_DrProb(X_mm).unsqueeze(-1)
            X_nn = normalize(X_mm, fun_q.bounds)
            # print(X_nn)
       
            # Y_next = torch.tensor(
            #     [eval_objective(x) for x in X_next], dtype=dtype, device=device
            # ).unsqueeze(-1)

            # Update state
            state = update_state(state=state, Y_next=Y_next)
                
            # Append data
            X_q = torch.cat((X_q, X_nn), dim=0)
            Y_turbo_q = torch.cat((Y_turbo_q, Y_next), dim=0)
            # print(X_q)
            torch.cuda.empty_cache()
            display.clear_output(wait=True)
        
            best_index = torch.eq(Y_turbo_q.squeeze(-1),state.best_value).nonzero().squeeze(-1)[0]
            predicted_parameters = unnormalize(X_q[best_index], fun_q.bounds)
            # predicted_parameters = X_turbo_q[best_index]

            print('\n--------------------------------------------\n') 
            print(f"{len(X_q)}) X_best: {unnormalize(X_q[best_index], fun_q.bounds)}, Best value: {state.best_value:.2e}, TR length: {state.length:.4e}")
            best_values.append(state.best_value)
            iteration_times.append((time.time() - start_time) / 60)
            

        Y_top = torch.argmax(Y_turbo_q)
        print(f"'X_best': {unnormalize(X_q[Y_top], fun_q.bounds)}")
        X_top = unnormalize(X_q[Y_top], fun_q.bounds).cpu().numpy()

    
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
    finally:
        Y_top = torch.argmax(Y_turbo_q)
        final_ram = process.memory_info().rss / (1024 * 1024)  # in MB
        ram_usage.append(final_ram)

        iterations = list(range(1, len(best_values) + 1))
        end_time = time.time()
        execution_time = end_time - start_time
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, [-value for value in best_values], marker='o', linestyle='-')
        plt.xlabel('Iteration')
        plt.ylabel('Best Value')
        plt.title('Convergence of TuRBO Model')
        plt.grid(True)
        plt.savefig('/home/phD/xiankang/drprobe_interface-master/drprobe_interface-master/examples/STO/time/column6/t01/convergence_plot_lgpn_f.png')
        plt.show()
        plt.close 
        
        plt.figure(figsize=(10, 6))
        plt.plot(iteration_times, [-value for value in best_values], marker='o', linestyle='-')
        plt.title('Best Value Progression Over Time')
        plt.xlabel('Time (minutes)')
        plt.ylabel('Best Value')
        plt.grid(True)
        plt.savefig('/home/phD/xiankang/drprobe_interface-master/drprobe_interface-master/examples/STO/time/column6/t01/best_values_plot.png')
        plt.show()       
        #return {'X_best': unnormalize(X_turbo_q[Y_top], fun_q.bounds), 'Best_value': state.best_value}

        data = pd.DataFrame({
            'Time (minutes)': iteration_times,
            'Best Value': [-value for value in best_values]
        })
        csv_path = '/home/phD/xiankang/drprobe_interface-master/drprobe_interface-master/examples/STO/time/column6/t01/best_values.csv'
        data.to_csv(csv_path, index=False)
        
        return {'X_best': unnormalize(X_q[Y_top], fun_q.bounds), 'Best_value': torch.max(Y_turbo_q)}

### Run the code
os.chdir(root_path)

io = IO(root_path)

if os.path.isfile('/home/phD/xiankang/drprobe_interface-master/drprobe_interface-master/examples/STO/time/column6/t01/new_resulst.txt'):
    restart = True
else:
    restart = False    

print(restart)    
if restart is False:

    result = BO_search_DrProb(acqf = 'qucb', length_min=0.1**5, hpar=10)
    stop_value = result['Best_value'].cpu().numpy()
 
    para_init = int(os.popen('cat '+root_path+folder+'/column6/t01/'+'new_resulst.txt | wc -l').read())+1
    with open('/home/phD/xiankang/drprobe_interface-master/drprobe_interface-master/examples/STO/time/column6/t01/new_resulst.txt', 'a') as f:
        f.write(str(para_init)+'   '+'_'.join([str(i.item()) for i in result['X_best']])+'   '+str(result['Best_value'].cpu().numpy())+'\n')
    # read_quanty_para(root_path=root_path, file='new_resulst.txt')
    display.clear_output(wait=True)    
n=0
while n<1:
    para_num = int(os.popen('cat '+root_path+folder+'/column6/t01/'+'new_resulst.txt | wc -l').read())+1
    print('parameter_num: ', para_num)
        
    result = BO_search_DrProb(acqf = 'qucb', length_min=0.1**5, hpar=5000)
    stop_value = result['Best_value'].cpu().numpy()
    print('stop_value_accurate: ', stop_value)
    para_init = int(os.popen('cat '+root_path+folder+'/'+'column6/t01/'+'new_resulst.txt | wc -l').read())+1
    with open('/home/phD/xiankang/drprobe_interface-master/drprobe_interface-master/examples/STO/time/column6/t01/new_resulst.txt', 'a') as f:
        f.write(str(para_init)+'   '+'_'.join([str(i.item()) for i in result['X_best']])+'   '+str(result['Best_value'].cpu().numpy())+'   No_restart\n')
    # read_quanty_para(root_path=root_path, file='new_resulst.txt')
    display.clear_output(wait=True)
    n += 1