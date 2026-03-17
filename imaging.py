import os, shutil
import pickle
import traceback
import math
from dataclasses import dataclass
from dataclasses import field
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
from botorch.optim import optimize_acqf, optimize_acqf_discrete
from botorch.utils.transforms import normalize
from botorch.utils.transforms import unnormalize
from torch.quasirandom import SobolEngine
from scipy.ndimage import gaussian_filter
import pandas as pd
import time
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
from sklearn.linear_model import LinearRegression
from __future__ import annotations
from abc import ABC
import psutil
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
torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double
print(device)
SMOKE_TEST = os.environ.get("SMOKE_TEST")
root_path = os.getcwd()+'/'

class IO(object):
    
    def __init__(self, root_path, dr_input_file = 'BTO110-2.cel', dr_input_para_file = 'inputpara.txt', dr_output_para_file = 'outputpara.txt',
                 dr_output_file_name='out.dat', real_data = 'K6.dat'):
    
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
folder = 'STO02'
io = IO(root_path)
input_file = io.read_input_file(folder)
a, b, c = np.genfromtxt(input_file, skip_header=1, skip_footer=1, usecols=(1, 2, 3))[0]
nx, ny = 128, 90
nz = 4
img_real = io.read_real_data(folder,45,64)
kl = img_real
plt.imshow(-kl)
plt.colorbar()
plt.show()
y_real_p = torch.tensor(kl, device = device, dtype = dtype)
y_real_n = torch.tensor(-kl, device = device, dtype = dtype)

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
        cel_file = io.read_input_file(folder)
        slice_name = root_path+folder+'/imaparafitting/'+'t01/task/slc/STO'
        msa_prm = root_path+folder+'/imaparafitting/'+ 't01/task/msa.prm'
        
        # para = io.read_parameters(folder)
        
        if X.ndim <= 1:
            X = X.unsqueeze(0)
        for folder_num, x in enumerate(X):
            Num_x = [i.item() for i in x]
            
            abf = Num_x[0]
            tilt_x = Num_x[1]
            tilt_y = Num_x[2]       
            vibration = Num_x[3]
            
            vibration_matrix = (1,vibration,vibration,0)
            A0_x = Num_x[4]
            A0_y = Num_x[5]
            defocus = Num_x[6] #C1
            two_fold_astigmatism_x = Num_x[7] #A1
            two_fold_astigmatism_y = Num_x[8]
            coma_x = Num_x[9] #B2
            coma_y = Num_x[10]
            three_fold_astigmatism_x = Num_x[11] #A2
            three_fold_astigmatism_y = Num_x[12]
            cs = Num_x[13] #C3
            CS = round(cs)*100
            star_aberration_x = Num_x[14]*1000 #S3
            star_aberration_y = Num_x[15]*1000
            four_fold_astigmatism_x = Num_x[16]*1000 #A3
            four_fold_astigmatism_y = Num_x[17]*1000
            c5 =  Num_x[18]
            C5 = round(c5)*1000000
            thick  = Num_x[19]
            print(thick)
            focus_spread = Num_x[20]
            semi_convergence = Num_x[21]
            
            ht = 300
            wlkev = 1.2398419843320025  # c * h / q_e * 1E6
            twomc2 = 1021.9978999923284  # 2*m0*c**2 / q_e
            
            STF_HT2WL = wlkev / math.sqrt(ht * (twomc2 + ht))

            # aberrations_dict = {1:(defocus,0),2:(two_fold_astigmatism_x,two_fold_astigmatism_y),3:(coma_x,coma_y),
            #             4:(three_f,old_astigmatism_x,three_fold_astigmatism_y),5:(CS,0),
            #                    6:(star_aberration_x,star_aberration_y),7:(four_fold_astigmatism_x,four_fold_astigmatism_y), 11:(4000000,0)}
            aberrations_dict = {0:(A0_x,A0_y), 1:(defocus,0),2:(two_fold_astigmatism_x,two_fold_astigmatism_y),3:(coma_x,coma_y),
                        4:(three_fold_astigmatism_x,three_fold_astigmatism_y),5:(CS,0),
                                6:(star_aberration_x,star_aberration_y),7:(four_fold_astigmatism_x,four_fold_astigmatism_y), 11:(C5,0)}
            # aberrations_dict = {1: (3.8, 0), 2:(1.5, 0), 3:(10.26, 28.19), 4:(5.21, 29.54), 5:(-15000, 0), 11:(4000000,0)}
        
            wave_output = root_path+folder+'/imaparafitting/'+'t01/task/wav/STO.wav'
            wavimg_prm = root_path+folder+'/imaparafitting/'+'t01/task/wavimg.prm'
        
            drp.commands.celslc(cel_file=cel_file, # location of cel file
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
            msa. conv_semi_angle = 0.2
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
            msa.tot_number_of_slices = 52 # Corresponds to 5 layers / unit cells of SrTiO3
            # Save prm file
            msa.save_msa_prm(msa_prm)
        
            drp.commands.msa(prm_file = msa_prm, 
                 output_file= wave_output,
                 ctem=True, # Flag for conventional TEM simulation (otherwise calculates STEM images)
                 output=True)
            wavimg = drp.WavimgPrm()
            sl = round(thick) # Perform simulation for slice # 2
            print(sl)
        
            wave_files = root_path+folder+'/imaparafitting/'+'t01/task/wav/STO_sl{:03d}.wav'.format(sl)
            output_files = root_path+folder+'/imaparafitting/'+'t01/task/img/STO_sl{:03d}.dat'.format(sl)
            stl = '{:03d}'.format(sl)
            fake_data = root_path+folder+'/imaparafitting/'+'t01/task/img/'+'STO_sl'+stl+'.dat'
            # Setup (fundamental) MSA parameters
            wavimg.high_tension = ht
            wavimg.mtf = (1, 1.068, 'MTF-US2k-300.mtf')  
            wavimg.oa_radius = 250 # Apply objective aperture [mrad]
            wavimg.oa_position = (0, 0)
            # wavimg.vibration = vibration_matrix # Apply isotropic image spread of 16 pm rms displacement
            wavimg.vibration = vibration_matrix
            wavimg.spat_coherence = (1, semi_convergence)
            wavimg.temp_coherence = (1, focus_spread) # Apply spatial coherence
            wavimg.wave_sampling = (a / nx, b / ny)
            wavimg.wave_files = wave_files          

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

            y_fake_p = torch.tensor(img_fake, device = device, dtype = dtype)
            y_fake_n = torch.tensor(-img_fake, device = device, dtype = dtype)
            
            Loss = -F.mse_loss(y_real_p,y_fake_p)
            loss.append(Loss) 
            display.clear_output(wait=True)
            
        return torch.tensor(loss, dtype=dtype, device=device)
def eval_objective_DrProb(x):
    """This is a helper function we use to unnormalize and evalaute a point"""
    return fun_q(unnormalize(x, fun_q.bounds))

fun_q = DrProbe(dim=22, negate=False).to(dtype=dtype, device=device) 
fun_q.bounds[0, :] = torch.tensor([0.01,-0.573, -0.573, 0.015, -0.05, -0.05, 0, -5, -5, -50,  -50, -50, -50, -150, -10, -10,-10, -10, 2,  1,  2, 0.1])
fun_q.bounds[1, :] = torch.tensor([0.05, 0.573,  0.573, 0.024,  0.05,  0.05, 10, 5,  5,  50,   50,  50,  50, -50,   10,  10, 10,  10, 10, 50, 4, 0.5])
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

    with open(root_path+folder+'/imaparafitting/'+'t01/'+'para_results.txt') as f:
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
        
        X_next, acq_value = optimize_acqf(
            ucb,
            bounds=torch.stack([tr_lb, tr_ub]),
            q=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )

    return {'X_next': X_next, 'acq':acq_value}

def BO_search_DrProb(acqf = 'qucb', length_min=0.1**7, hpar=500):
    batch_size = 8
    n_init = 4 * dim_q
    
    if restart is False:        
        X_turbo_q = get_initial_points(dim_q, n_init)
        Y_turbo_q = eval_objective_DrProb(X_turbo_q).unsqueeze(-1)
    else:
        turbo_info = restarts_BO_q(root_path)
        X_turbo_q_0 = turbo_info['x']
        X_turbo_q_1 = get_initial_points(dim_q, n_init)
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
    start_time = time.time()
    process = psutil.Process()

    mae_per_parameter = {'abf': [], 'tilt_x': [],  'tilt_y': [], 'vibration': [],  'defocus': [],  'two_fold_astigmatism_x': [],
                     'two_fold_astigmatism_y': [], 'coma_x': [],  'coma_y': [], 'three_fold_astigmatism_x':[], 'three_fold_astigmatism_y':[],
                    'CS': [],'thickness':[]}
                    
    # mae_per_parameter = {'abf': [], 'tilt_x': [],  'tilt_y': []}
  

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
            torch.cuda.empty_cache()
       
            # Y_next = torch.tensor(
            #     [eval_objective(x) for x in X_next], dtype=dtype, device=device
            # ).unsqueeze(-1)

            # Update state
            state = update_state(state=state, Y_next=Y_next)
                
            # Append data
            X_turbo_q = torch.cat((X_turbo_q, X_next), dim=0)
            Y_turbo_q = torch.cat((Y_turbo_q, Y_next), dim=0)
        
            best_index = torch.eq(Y_turbo_q.squeeze(-1),state.best_value).nonzero().squeeze(-1)[0]
            display.clear_output(wait=True)
        
            # img_path = 'pixel'+str(len(X_turbo_q))
        
            # BF_image(unnormalize(X_turbo_q[best_index], fun_q.bounds), img_path = img_path)  
        
            predicted_parameters = unnormalize(X_turbo_q[best_index], fun_q.bounds)

            print('\n--------------------------------------------\n') 
            print(f"{len(X_turbo_q)}) X_best: {unnormalize(X_turbo_q[best_index], fun_q.bounds)}, Best value: {state.best_value:.2e}, TR length: {state.length:.4e}")
            best_values.append(state.best_value)
            iteration_times.append((time.time() - start_time) / 60)

        Y_top = torch.argmax(Y_turbo_q)
        print(f"'X_best': {unnormalize(X_turbo_q[Y_top], fun_q.bounds)}")
        X_top = unnormalize(X_turbo_q[Y_top], fun_q.bounds).cpu().numpy()

        parameter_results = {'abf': X_top[0], 'tilt_x':X_top[1],
                                            'tilt_y':X_top[2],
                                            'vibration':(1,X_top[3],X_top[3],0),
                                             'defocus':(X_top[4],0),
                                             '2-fold-astigmatism':(X_top[5],X_top[6]),
                                             'coma':(X_top[7],X_top[8]),
                                            'three_fold_astigmatism': (X_top[9],X_top[10]),
                                            'CS': (X_top[11],0), 'thickness':(int(X_top[12]))}

    
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
    finally:
        Y_top = torch.argmax(Y_turbo_q)
    
        iterations = list(range(1, len(best_values) + 1))
        end_time = time.time()
        execution_time = end_time - start_time
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, [-value for value in best_values], marker='o', linestyle='-')
        plt.xlabel('Iteration')
        plt.ylabel('Best Value')
        plt.title('Convergence of TuRBO Model')
        plt.grid(True)
        plt.savefig('/home/phD/xiankang/drprobe_interface-master/drprobe_interface-master/examples/STO/STO02/imaparafitting/t01/convergence_plot_lgpn_f.png')
        plt.show()
        plt.close 
        
        plt.figure(figsize=(10, 6))
        plt.plot(iteration_times, [-value for value in best_values], marker='o', linestyle='-')
        plt.title('Best Value Progression Over Time')
        plt.xlabel('Time (minutes)')
        plt.ylabel('Best Value')
        plt.grid(True)
        plt.savefig('/home/phD/xiankang/drprobe_interface-master/drprobe_interface-master/examples/STO/STO02/imaparafitting/t01/best_values_plot.png')
        plt.show()       
        df = pd.DataFrame({
            'Time (minutes)': iteration_times,
            'Best Value': [-value for value in best_values]
        })
        
        csv_path = "/home/phD/xiankang/drprobe_interface-master/drprobe_interface-master/examples/STO/STO02/imaparafitting/t01/best_values.csv"
        df.to_csv(csv_path, index=False)
        
        return {'X_best': unnormalize(X_turbo_q[Y_top], fun_q.bounds), 'Best_value': torch.max(Y_turbo_q)}
os.chdir(root_path)

io = IO(root_path)

if os.path.isfile('/home/phD/xiankang/drprobe_interface-master/drprobe_interface-master/examples/STO/STO02/imaparafitting/t01/para_results.txt'):
    restart = True
else:
    restart = False    
        
# if initialize is False:
#     _X_data = Candidate_X[numpy.random.randint(0, len(Candidate_X), 20)]
#     print('initialization_data: ', _X_data)
#     X = torch.tensor(_X_data, dtype=dtype, device=device)
#     Y = get_RIXS(unnormalize(X, fun_X.bounds))
#     initialize = True
print(restart)    
if restart is False:

    result = BO_search_DrProb(acqf = 'qucb', length_min=0.1**7, hpar=0.25)
    stop_value = result['Best_value'].cpu().numpy()
 
    para_init = int(os.popen('cat '+root_path+folder+'/imaparafitting/t01/'+'para_results.txt | wc -l').read())+1
    with open('/home/phD/xiankang/drprobe_interface-master/drprobe_interface-master/examples/STO/STO02/imaparafitting/t01/para_results.txt', 'a') as f:
        f.write(str(para_init)+'   '+'_'.join([str(i.item()) for i in result['X_best']])+'   '+str(result['Best_value'].cpu().numpy())+'\n')
    # read_quanty_para(root_path=root_path, file='para_results.txt')
    display.clear_output(wait=True)    
n=0
while n<20:
    para_num = int(os.popen('cat '+root_path+folder+'/t01/'+'para_results.txt | wc -l').read())+1
    print('parameter_num: ', para_num)
        
    result = BO_search_DrProb(acqf = 'qucb', length_min=0.1**5, hpar=500)
    stop_value = result['Best_value'].cpu().numpy()

    print('stop_value_accurate: ', stop_value)

    para_init = int(os.popen('cat '+root_path+folder+'/imaparafitting/'+'t01/'+'para_results.txt | wc -l').read())+1
    with open('/home/phD/xiankang/drprobe_interface-master/drprobe_interface-master/examples/STO/STO02/imaparafitting/t01/para_results.txt', 'a') as f:
        f.write(str(para_init)+'   '+'_'.join([str(i.item()) for i in result['X_best']])+'   '+str(result['Best_value'].cpu().numpy())+'   No_restart\n')
    # read_quanty_para(root_path=root_path, file='para_results.txt')
    display.clear_output(wait=True)
    n += 1
