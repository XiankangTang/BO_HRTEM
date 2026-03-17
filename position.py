import os, shutil
import pickle
import traceback
import math
from dataclasses import dataclass
from dataclasses import field
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import psutil
from botorch.acquisition import qExpectedImprovement
from botorch.acquisition import qUpperConfidenceBound
from botorch.acquisition import UpperConfidenceBound
from botorch.fit import fit_gpytorch_model
from botorch.generation import MaxPosteriorSampling
from botorch.models import FixedNoiseGP, SingleTaskGP, HeteroskedasticSingleTaskGP
from botorch.optim import optimize_acqf, optimize_acqf_discrete
from botorch.utils.transforms import normalize
from botorch.utils.transforms import unnormalize
from torch.quasirandom import SobolEngine
import gpytorch
from gpytorch.constraints import Interval
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import HorseshoePrior
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel, RBFKernelGrad, SpectralDeltaKernel, SpectralMixtureKernel
from scipy.signal import find_peaks
from typing import List, Optional, Tuple
from botorch.test_functions.base import BaseTestProblem
from torch import Tensor
import time
from __future__ import annotations
from abc import ABC
from copy import deepcopy
from typing import Dict, Optional, Tuple, Union
from botorch.test_functions.synthetic import SyntheticTestFunction
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.exceptions import UnsupportedError
from botorch.models.gp_regression import FixedNoiseGP
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
from PIL import Image
from scipy.ndimage import maximum_filter
import IPython.display as display
import subprocess

torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# dtype = torch.float
dtype = torch.double
print(device)
SMOKE_TEST = os.environ.get("SMOKE_TEST")

# os.system('export OMP_NUM_THREADS=48')
# os.system('ulimit -v 48000000')
root_path = os.getcwd()+'/'

def distance_to_other_points(point, points):  ## calculate the bond length
    distances = []
    for other_point in points:
        if other_point != point:
            distance = torch.sqrt((point[0] - other_point[0]) ** 2 + (point[1] - other_point[1]) ** 2 + (point[2] - other_point[2]) ** 2)
            distances.append(distance.item())
    return distances

def cal_bond(X_init, path):
    
    for i in range(X_init.shape[0]):
        X = unnormalize(X_init[i], fun_q.bounds)
        cat = []
        a = np.genfromtxt(path, skip_header=2, skip_footer=1, usecols=(1, 2, 3))
        b = a[:,-1]

        for j in range(0, len(X), 2):
            cat.append((X[j], X[j+1]))
        tensor_to_concat = torch.tensor(b).to(dtype=dtype, device=device)
        c_coordinates = [(coord[0], coord[1], elem) for coord, elem in zip(cat, tensor_to_concat)] 

        for j in range(len(c_coordinates)):
            specific_point = c_coordinates[j]
            distances = distance_to_other_points(specific_point, c_coordinates[:j] + c_coordinates[j + 1:])
            if any(element <= 0 for element in distances):
                X_init = X_init[:i] + X_init[i + 1:] 
        return X_init
    
def cal_bond2(X_init,acq_value, path):
    
    for i in range(X_init.shape[0]):
        X = unnormalize(X_init[i], fun_q.bounds)
        cat = []
        a = np.genfromtxt(path, skip_header=2, skip_footer=1, usecols=(1, 2, 3))
        b = a[:,-1]

        for j in range(0, len(X), 2):
            cat.append((X[j], X[j+1]))
        tensor_to_concat = torch.tensor(b).to(dtype=dtype, device=device)
        c_coordinates = [(coord[0], coord[1], elem) for coord, elem in zip(cat, tensor_to_concat)] 

        for j in range(len(c_coordinates)):
            specific_point = c_coordinates[j]
            distances = distance_to_other_points(specific_point, c_coordinates[:j] + c_coordinates[j + 1:])
            if any(element <= 0 for element in distances):
                X_init = X_init[:i] + X_init[i + 1:] 
                acq_value = acq_value[:i]+acq_value[i+1:]
        return {'X_next': X_init, 'acq_value':acq_value}

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
        co = subprocess.check_output('/home/phD/xiankang/anaconda3/envs/drprobe/lib/python3.11/site-packages/drprobe/'+_command, shell=True)
        print('Performed BuildCell with the following command:\n','/home/phD/xiankang/anaconda3/envs/drprobe/lib/python3.11/site-packages/drprobe/' +_command)
        print(co.decode('utf-8'))
    else:
        subprocess.call(_command, shell=True)

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        # If the folder does not exist, create it
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")
        
def map_to_interval(lst):
    mapped_list = []
    for num in lst:
        if num < 0:
            mapped_list.append(0)
        elif num > 1:
            mapped_list.append(1)
        else:
            mapped_list.append(num)
    return mapped_list

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
        slice_name = root_path+folder+'/'+'position/tave/task/slc/BTO'
        msa_prm = root_path+folder+'/'+ 'position/tave/task/msa.prm'
        output_path= root_path+folder+'/'+'position/tave/'
        output_file = output_path+'BTO.cel'
        
        if X.ndim <= 1:
            X = X.unsqueeze(0)
        for folder_num, x in enumerate(X):
            Num_x = [i.item() for i in x]
            x0 = Num_x[0]
            y0 = Num_x[1]
            x1 = Num_x[2]
            y1 = Num_x[3]
            x2 = Num_x[4]
            y2 = Num_x[5]
            x3 = Num_x[6]
            y3 = Num_x[7]
            x4 = Num_x[8]
            y4 = Num_x[9]
            x5 = Num_x[10]
            y5 = Num_x[11]
            
            atoms = {'Ba':[(x0,y0, 0.875, 1., 0.4272),(x1,y1, 0.375, 1., 0.4272)],
             'Ti':[(x2,y2,0.875,1.0,0.6948),(x3,y3,0.375, 1.0,0.6948)],
             'O':[(x1, y1, 0.875000, 1.0, 0.5611), (x0, y0, 0.375, 1.0, 0.5611),(x4, y4, 0.125, 1.0, 0.5611), (x5, y5, 0.625, 1.0, 0.5611),
                  (x5, y5, 0.125, 1.0, 0.5611),(x4, y4, 0.625, 1.0, 0.5611)]}

            BuildCell(output_file = output_file,lattice=lattice, atoms = atoms, override=True, output = True)
                        
            # a, b, c = np.genfromtxt(output_file, skip_header=1, skip_footer=1, usecols=(1, 2, 3))[0]
            # nx, ny = 64, 45
            # nz = 4          
            
            abf = 0.018665780196320564
            tilt_x = 0.20664366500790854
            tilt_y = 0.21329664140786198
            vibration = 0.018
            vibration_matrix = (1,vibration,vibration,0)
            thick  = 20         
            ht = 300
            wlkev = 1.2398419843320025  # c * h / q_e * 1E6
            twomc2 = 1021.9978999923284  # 2*m0*c**2 / q_e
            
            STF_HT2WL = wlkev / math.sqrt(ht * (twomc2 + ht))

            aberrations_dict = {1: (4.580276748171563,0.0000), 2:(-1.2859117804371323,2.148040446200495), 3:(-82.0447946299991,38.970930108623435), 
                          4:(62.05326169583384,-19.27460858682292),5:(-5000,0),6:(1999.9577,-2000.0000),7:(-1172.4281,-1999.9574), 11:(2000000,0)}
        
            wave_output = root_path+folder+'/'+'position/tave/task/wav/BTO.wav'
            wavimg_prm = root_path+folder+'/'+'position/tave/task/wavimg.prm'
        
            drp.commands.celslc(cel_file=output_file, # location of cel file
                    slice_name=slice_name, # target file names
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
            msa.number_of_slices = 4 # Number of slices in phase gratings
            msa.det_readout_period = 1 # Readout after every 2 slices
            msa.tot_number_of_slices = 20 # Corresponds to 5 layers / unit cells of SrTiO3
            # Save prm file
            msa.save_msa_prm(msa_prm)
        
            drp.commands.msa(prm_file = msa_prm, 
                 output_file= wave_output,
                 ctem=True, # Flag for conventional TEM simulation (otherwise calculates STEM images)
                 output=True)
            wavimg = drp.WavimgPrm()
            sl = round(thick) # Perform simulation for slice # 2
            print(sl)
        
            wave_files = root_path+folder+'/'+'position/tave/task/wav/BTO_sl{:03d}.wav'.format(sl)
            output_files = root_path+folder+'/'+'position/tave/task/img/BTO_sl{:03d}.dat'.format(sl)
            stl = '{:03d}'.format(sl)
            fake_data = root_path+folder+'/'+'position/tave/task/img/'+'BTO_sl'+stl+'.dat'
            # Setup (fundamental) MSA parameters
            wavimg.high_tension = ht
            wavimg.mtf = (1, 1.068, 'MTF-US2k-300.mtf')  
            wavimg.oa_radius = 250 # Apply objective aperture [mrad]
            wavimg.oa_position = (0, 0)
            # wavimg.vibration = vibration_matrix # Apply isotropic image spread of 16 pm rms displacement
            wavimg.vibration = vibration_matrix
            wavimg.spat_coherence = (1, 0.45523396943833877)
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

            loss.append(Loss) 
        
        return torch.tensor(loss, dtype=dtype, device=device)

def eval_objective_DrProb(x):
    """This is a helper function we use to unnormalize and evalaute a point"""
    return fun_q(unnormalize(x, fun_q.bounds))

def distance(point1, point2):
    """
    Calculate the Euclidean distance between two points.
    """
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

path = 'time/BTO110-2.cel'
fun_q = DrProbe(dim=12, negate=False).to(dtype=dtype, device=device) 
cc = np.genfromtxt(path, skip_header=2, skip_footer=1, usecols=(1, 2, 3))
cat = []
for i in range(8):
    if i == 4 or i == 5:  
        continue
    else:
        X = cc[i]
        cat.append([X[0], X[1]])
unique_list = []
seen = set()
for item in cat:
    # Convert list to tuple to make it hashable (for set)
    t = tuple(item)
    if t not in seen:
        unique_list.append(item)
        seen.add(t)
min_distance = float('inf')  # Initialize with positive infinity
for i in range(len(unique_list)):
    for j in range(i+1, len(unique_list)):
        dist = distance(unique_list[i], unique_list[j])
        min_distance = min(min_distance, dist)        
print(min_distance)
flattened_list = [item for sublist in cat for item in sublist]
b1 = [x - 0.5*min_distance for x in flattened_list]
b2 = [x + 0.5*min_distance for x in flattened_list]
b1 = map_to_interval(b1)
b2 = map_to_interval(b2)
fun_q.bounds[0, :] = torch.tensor(b1).to(dtype=dtype, device=device) 
fun_q.bounds[1, :] = torch.tensor(b2).to(dtype=dtype, device=device) 
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

state_q = TurboState(dim = dim_q, batch_size = 4)
print(state_q)

def get_initial_points(dim, n_pts, path):
    #sobol = SobolEngine(dimension=dim, scramble=True, seed=0)
    sobol = SobolEngine(dimension=dim, scramble=True)
    X_init = sobol.draw(n=n_pts).to(dtype=dtype, device=device)
    X_l = cal_bond(X_init, path)
    while n_pts > X_l.shape[0]:
        n = n_pts - X_l.shape[0]
        X_new = sobol.draw(n=n).to(dtype=dtype, device=device)
        X_nl = cal_bond(X_l, path)
        X_l = torch.cat((X_l, X_nl), dim=0)
    return X_l

def restarts_BO_q(root_path=root_path):

    with open(root_path+folder+'/'+'position/tave/'+'atom_results.txt') as f:
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
            
        X = cal_bond2(X_next, acq_value, path)
        X_n = X['X_next']
        acq_value = X['acq_value']
        while batch_size > X_n.shape[0]:
            n = batch_size - X_n.shape[0]
            X_new, nacq_value = optimize_acqf(
                qucb,
                bounds=torch.stack([tr_lb, tr_ub]),
                q=n,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
            )        
            x_nl = cal_bond2(X_new, nacq_value, path)
            X_n = torch.cat((X_n, X_nl), dim=0)
            acq_value = torch.cat((acq_value, nacq_value), dim=0)
        X_next = X_n
        
    elif acqf == "ucb":
        ucb = UpperConfidenceBound(model, beta = hpar, record = record)
        
        X_next, acq_value = optimize_acqf(
            ucb,
            bounds=torch.stack([tr_lb, tr_ub]),
            q=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )

    return {'X_next': X_next, 'acq':acq_value}

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

def BO_search_DrProb(acqf = 'qucb', length_min=0.1**4, hpar=500):
    batch_size = 8
    n_init = 4 * dim_q
    
    start_time = time.time()
    process = psutil.Process()
    
    # Get initial RAM usage
    initial_ram = process.memory_info().rss / (1024 * 1024)  # in MB
    ram_usage = [initial_ram]
    
    if restart is False:        
        X_turbo_q = get_initial_points(dim_q, n_init, path = 'time/BTO110-2.cel')
        Y_turbo_q = eval_objective_DrProb(X_turbo_q).unsqueeze(-1)
    else:
        turbo_info = restarts_BO_q(root_path)
        X_turbo_q_0 = turbo_info['x']
        X_turbo_q_1 = get_initial_points(dim_q, n_init,path = 'time/BTO110-2.cel')
        X_turbo_q = torch.cat((X_turbo_q_0, X_turbo_q_1), dim=0)
        print('number_of_sample: ', len(X_turbo_q))
        Y_turbo_q = eval_objective_DrProb(X_turbo_q).unsqueeze(-1)
        
    state = TurboState(dim_q, batch_size=batch_size)

    NUM_RESTARTS = 10 if not SMOKE_TEST else 2
    RAW_SAMPLES = 16384 if not SMOKE_TEST else 4
    N_CANDIDATES = min(5000, max(2000, 200 * dim_q)) if not SMOKE_TEST else 4
    parameter_results = {}
    torch.manual_seed(0)
    best_values = [] 
    iteration_times = [] 
  

    try:
        while not state.restart_triggered:  # Run until TuRBO converges
            # Fit a GP model
            train_Y_q = (Y_turbo_q - Y_turbo_q.mean()) / Y_turbo_q.std()
        
            likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
            covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
                MaternKernel(
                    nu=2.5, ard_num_dims=dim_q, lengthscale_constraint=Interval(0.005, 4.0)
                )
            )

            # covar_module = ScaleKernel(SpectralDeltaKernel(num_dims=dim_q, num_deltas=32))

            train_Yvar_q = torch.full_like(train_Y_q, 0.05)
            model = FixedNoiseGP(X_turbo_q, train_Y_q, train_Yvar_q, covar_module=covar_module)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)

            fit_gpytorch_model(mll)
            acqf = 'qucb'

            # Create a batch
            X_info = generate_batch(
                state=state,
                model=model,
                X=X_turbo_q,
                Y=train_Y_q,
                batch_size=batch_size,
                n_candidates=N_CANDIDATES,
                num_restarts=NUM_RESTARTS,
                raw_samples=RAW_SAMPLES,
                acqf=acqf,
                hpar=hpar,
                )
            X_next = X_info['X_next']
            Y_next = eval_objective_DrProb(X_next).unsqueeze(-1)
            display.clear_output(wait=True)

            # Y_next = torch.tensor(
            #     [eval_objective(x) for x in X_next], dtype=dtype, device=device
            # ).unsqueeze(-1)

            # Update state
            state = update_state(state=state, Y_next=Y_next)
                
            # Append data
            X_turbo_q = torch.cat((X_turbo_q, X_next), dim=0)
            Y_turbo_q = torch.cat((Y_turbo_q, Y_next), dim=0)
        
            best_index = torch.eq(Y_turbo_q.squeeze(-1),state.best_value).nonzero().squeeze(-1)[0]
         
        
            predicted_parameters = unnormalize(X_turbo_q[best_index], fun_q.bounds)

            print('\n--------------------------------------------\n') 
            print(f"{len(X_turbo_q)}) X_best: {unnormalize(X_turbo_q[best_index], fun_q.bounds)}, Best value: {state.best_value:.2e}, TR length: {state.length:.4e}")
            best_values.append(state.best_value)
            iteration_times.append((time.time() - start_time) / 60)
            torch.cuda.empty_cache()
            display.clear_output(wait=True)

        Y_top = torch.argmax(Y_turbo_q)
        print(f"'X_best': {unnormalize(X_turbo_q[Y_top], fun_q.bounds)}")
        X_top = unnormalize(X_turbo_q[Y_top], fun_q.bounds).cpu().numpy()
        
    
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
    finally:
        Y_top = torch.argmax(Y_turbo_q)
        final_ram = process.memory_info().rss / (1024 * 1024)  # in MB
        ram_usage.append(final_ram)

        end_time = time.time()
        execution_time = end_time - start_time

        iterations = list(range(1, len(best_values) + 1))
        end_time = time.time()
        execution_time = end_time - start_time
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, [-value for value in best_values], marker='o', linestyle='-')
        plt.xlabel('Iteration')
        plt.ylabel('Best Value')
        plt.title('Convergence of TuRBO Model')
        plt.grid(True)
        plt.savefig('/home/phD/xiankang/drprobe_interface-master/drprobe_interface-master/examples/STO/time/position/tave/convergence_plot_lgpn_f.png')
        plt.show()
        plt.close 
        
        plt.figure(figsize=(10, 6))
        plt.plot(iteration_times, [-value for value in best_values], marker='o', linestyle='-')
        plt.title('Best Value Progression Over Time')
        plt.xlabel('Time (minutes)')
        plt.ylabel('Best Value')
        plt.grid(True)
        plt.savefig('/home/phD/xiankang/drprobe_interface-master/drprobe_interface-master/examples/STO/time/position/tave/best_values_plot.png')
        plt.show()       
        data = pd.DataFrame({
            'Time (minutes)': iteration_times,
            'Best Value': [-value for value in best_values]
        })
        csv_path = '/home/phD/xiankang/drprobe_interface-master/drprobe_interface-master/examples/STO/time/position/tave/best_values.csv'
        data.to_csv(csv_path, index=False)
        #return {'X_best': unnormalize(X_turbo_q[Y_top], fun_q.bounds), 'Best_value': state.best_value}
        return {'X_best': unnormalize(X_turbo_q[Y_top], fun_q.bounds), 'Best_value': torch.max(Y_turbo_q)}

### Run the code
os.chdir(root_path)

io = IO(root_path)

if os.path.isfile('/home/phD/xiankang/drprobe_interface-master/drprobe_interface-master/examples/STO/time/position/tave/atom_results.txt'):
    restart = True
else:
    restart = False    
        
if restart is False:
    result = BO_search_DrProb(acqf = 'qucb', length_min=0.1**7, hpar=0.25)
    stop_value = result['Best_value'].cpu().numpy()
    para_init = int(os.popen('cat '+root_path+folder+'/position/tave/'+'atom_results.txt | wc -l').read())+1
    with open('/home/phD/xiankang/drprobe_interface-master/drprobe_interface-master/examples/STO/time/position/tave/atom_results.txt', 'a') as f:
        f.write(str(para_init)+'   '+'_'.join([str(i.item()) for i in result['X_best']])+'   '+str(result['Best_value'].cpu().numpy())+'\n')
    # read_quanty_para(root_path=root_path, file='atom_results.txt')
    display.clear_output(wait=True)    
n=0
while n<0:
    para_num = int(os.popen('cat '+root_path+folder+'/position/tave/'+'atom_results.txt | wc -l').read())+1
    print('parameter_num: ', para_num)
        
    result = BO_search_DrProb(acqf = 'qucb', length_min=0.1**4, hpar=500)
    stop_value = result['Best_value'].cpu().numpy()

    print('stop_value_accurate: ', stop_value)

    para_init = int(os.popen('cat '+root_path+folder+'/'+'position/tave/'+'atom_results.txt | wc -l').read())+1
    with open('/home/phD/xiankang/drprobe_interface-master/drprobe_interface-master/examples/STO/time/position/tave/atom_results.txt', 'a') as f:
        f.write(str(para_init)+'   '+'_'.join([str(i.item()) for i in result['X_best']])+'   '+str(result['Best_value'].cpu().numpy())+'   No_restart\n')
    # read_quanty_para(root_path=root_path, file='atom_results.txt')
    display.clear_output(wait=True)
    n += 1