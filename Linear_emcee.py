#Import Libraries
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import emcee
from scipy.integrate import cumulative_trapezoid
from scipy.integrate import solve_ivp, quad
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from numpy.linalg import inv
from tqdm import tqdm
import scipy.linalg as la
from scipy import stats
import pandas as pd
from getdist import plots, MCSamples
from multiprocessing import Pool, cpu_count
import DESI_DR2

# ------------------------------------------------------------------------------
# Load and Prepare the Cosmic Chronometer (CC) Dataset for Likelihood Analysis
# ------------------------------------------------------------------------------

# This script utilizes the Cosmic Chronometers method to constrain cosmological 
# parameters using H(z) measurements obtained from passively evolving galaxies.

# --------------------------------------------------------------------------------
# Citing the Data and Methodology:
#
# If you use this dataset or analysis method in your work, please cite the following
# foundational studies by Prof. Dr. Moresco and collaborators:
#
# - Moresco et al. (2018), https://doi.org/10.48550/arXiv.1804.05864
# - Moresco et al. (2020), https://doi.org/10.48550/arXiv.2003.07362
#
# If you are using the same data points (as included in this script), please also cite:
#
# - Moresco et al. (2012), https://doi.org/10.48550/arXiv.1201.3609
# - Moresco (2015),       https://doi.org/10.48550/arXiv.1503.01116
# - Moresco et al. (2016), https://doi.org/10.48550/arXiv.1601.01701
#
# These studies provide the measurements and methodology for H(z) data based on the
# differential age evolution of early-type galaxies, which form the basis of the CC approach.
# --------------------------------------------------------------------------------

filename = 'data/HzTable_MM_BC03.dat'
z, Hz, errHz = np.genfromtxt(filename, comments='#', usecols=(0,1,2), unpack=True, delimiter=',')
ref = np.genfromtxt(filename, comments='#', usecols=(3), unpack=True, dtype=str, delimiter=',')

filename = 'data/data_MM20.dat'
zmod, imf, slib, sps, spsooo = np.genfromtxt(filename, comments='#', usecols=(0,1,2,3,4), unpack=True)

cov_mat_diag = np.zeros((len(z), len(z)), dtype='float64') 

for i in range(len(z)):
	cov_mat_diag[i,i] = errHz[i]**2

imf_intp = np.interp(z, zmod, imf)/100
slib_intp = np.interp(z, zmod, slib)/100
sps_intp = np.interp(z, zmod, sps)/100
spsooo_intp = np.interp(z, zmod, spsooo)/100

cov_mat_imf = np.zeros((len(z), len(z)), dtype='float64')
cov_mat_slib = np.zeros((len(z), len(z)), dtype='float64')
cov_mat_sps = np.zeros((len(z), len(z)), dtype='float64')
cov_mat_spsooo = np.zeros((len(z), len(z)), dtype='float64')

for i in range(len(z)):
	for j in range(len(z)):
		cov_mat_imf[i,j] = Hz[i] * imf_intp[i] * Hz[j] * imf_intp[j]
		cov_mat_slib[i,j] = Hz[i] * slib_intp[i] * Hz[j] * slib_intp[j]
		cov_mat_sps[i,j] = Hz[i] * sps_intp[i] * Hz[j] * sps_intp[j]
		cov_mat_spsooo[i,j] = Hz[i] * spsooo_intp[i] * Hz[j] * spsooo_intp[j]
          
cov_mat_cc = cov_mat_spsooo + cov_mat_imf + cov_mat_diag
inv_cov_mat = inv(cov_mat_cc)
cov_mat_cc = inv_cov_mat 

# ------------------------------------------------------------------------------
# Load and Prepare the Pantheon+ Dataset for Cosmological Likelihood Analysis
# ------------------------------------------------------------------------------

# This script processes the Pantheon+ Type Ia Supernova (SN Ia) dataset and prepares
# it for cosmological likelihood evaluation using a custom model or pipeline
# (e.g., emcee or cosmosis-based workflows).

# --------------------------------------------------------------------------------
# Citing the Data and Methodology:
#
# If you use this dataset or script in your work, please cite the following:
#
# - Pantheon+ compilation and analysis methodology:
#   Brout et al. (2022), https://doi.org/10.48550/arXiv.2112.03863
#   Scolnic et al. (2022), https://doi.org/10.48550/arXiv.2202.04077
#
# - For the construction of covariance matrices accounting for statistical and
#   systematic uncertainties and correlations in SN Ia light curves:
#   Conley et al. (2011), https://doi.org/10.48550/arXiv.1104.1443
#
# These references provide the foundation for the covariance matrix structure and
# likelihood computations used in this analysis.
# --------------------------------------------------------------------------------

values_filename = 'data/Pantheon+SH0ES.dat'
cov_filename = 'data/Pantheon+SH0ES_STAT+SYS.cov'

data = pd.read_csv(values_filename, sep=r'\s+')
origlen = len(data)
ww = (data['zHD'] > 0.01)  # Filter condition for zHD > 0.01
zcmb = data['zHD'][ww].values  # vpec corrected redshift (zCMB)
zhelio = data['zHEL'][ww].values  # Heliocentric redshift
mag = data['m_b_corr'][ww].values  # Corrected magnitudes
N = len(mag)

filename = cov_filename
#print("Loading covariance from {}".format(filename))
f = open(filename)
line = f.readline()
n = int(len(zcmb))
C = np.zeros((n,n))
ii = -1
jj = -1
mine = 999
maxe = -999
for i in range(origlen):
    jj = -1
    if ww[i]:
        ii += 1
    for j in range(origlen):
        if ww[j]:
            jj += 1
        val = float(f.readline())
        if ww[i]:
            if ww[j]:
                C[ii,jj] = val

f.close()
#print('Done')
cov = C
xdiag = 1/cov.diagonal()  # diagonal before marginalising constant
zmin = zcmb.min()
zmax = zcmb.max()
zmaxi = 1.1 ## we interpolate to 1.1 beyond that exact calc
#print("Pantheon SN: zmin=%f zmax=%f N=%i" % (zmin, zmax, N))
ninterp=150
zinter = np.linspace(1e-3, zmaxi, ninterp)
icov = la.inv(cov)

c = 2.99792458e5

def equation(z, y, params):

    Omega0 , Sigma0 , H0 , M , rd = params

    h, Omega  = y 
    dh_dz = (3 * h**2 - 3 * Sigma0 * (2 * h * Omega - Omega**2)) / (2 * (1 + z) * h)
    dOmega_dz = (-2 * (3 * Sigma0 - 2) * h * Omega - (1 - 3 * Sigma0) * Omega**2) / (2 * (1 + z) * h)
    
    return np.array([dh_dz, dOmega_dz]) 

def log_likelihood(params):

    Omega0 , Sigma0 , H0 , M , rd = params

    y_initial = [1, Omega0]
    sol = solve_ivp(lambda t, y: equation(t, y, params), [0, 3], y_initial, t_eval=np.linspace(0.0, 3, 800), rtol=1e-3, atol=1e-6, method='RK45')
    tsol = sol.t
	
    # Hubble-Chi2
    H_model = sol.y[0]

    # for any H_value 
    H_val = interp1d(tsol, H_model, kind='linear', fill_value="extrapolate")
    H_model_h11 = interp1d(tsol, H_model, kind='linear', fill_value="extrapolate")(z)
    H_model_h = H0 * H_model_h11
    res_hubble = H_model_h - Hz
    chi_h = -0.5 * (res_hubble.T @ cov_mat_cc @ res_hubble)
    
    # Vectorized integration using cumulative trapezoidal rule
    H_model_h2 = 1 / H_val(zinter)
    integral_grid = cumulative_trapezoid(H_model_h2, zinter, initial=0)
    integral = np.interp(zcmb, zinter, integral_grid)
    part1 = np.array([c * (1 + zi) / H0 for zi in zcmb])

    # SNIa-Chi2
    dL_model = part1 * integral
    m_model = 5 * np.log10(dL_model) + 25 + M
    residual = mag - m_model
    chi_p = -0.5 * np.dot(residual.T, np.dot(icov, residual))

    # DESI DR2 Chi2
    z_grid = np.linspace(0, 3, 800)
    H_model_bao = 1 / H_val(z_grid)  # Fix: use the interpolated H(z)
    integral_grid_bao = cumulative_trapezoid(H_model_bao, z_grid, initial=0)
    chiDESI = DESI_DR2.desidr2_likelihood(z_grid=z_grid, integral_grid=integral_grid_bao, H_val=H_val, params=params)
    
    chi = chi_h + chiDESI + chi_p
    
    return chi

def log_prior(params):
    
    Omega0 , Sigma0 , H0 , M , rd = params
    
    if not 0 < Omega0 < 0.50 :
        return -np.inf
    
    if not 0.6 < Sigma0 < 2.0 :
        return -np.inf
    
    if not 50.0 < H0 < 100.0:
        return -np.inf
    
    if not -20. < M < -18. :
        return -np.inf
    
    if not 100. < rd < 300.:
        return -np.inf
    
    return 0

def log_posterior(params, return_likelihood_only=False):
    prior = log_prior(params)

    if np.isinf(prior):
        return (-np.inf, None) if return_likelihood_only else -np.inf

    likelihood = log_likelihood(params)

    if np.isinf(likelihood):
        return (-np.inf, None) if return_likelihood_only else -np.inf

    if return_likelihood_only:
        return prior + likelihood, likelihood
    else:
        return prior + likelihood

def main():
    nsteps = 10000
    nwalkers = 30
    ndim = 5

    p0 = np.random.uniform(low=[0., 0.6, 50., -20., 100.], high=[0.50 , 2.0, 100., -18. ,300.], size=(nwalkers, ndim))

    print("Now we are computing")

    with Pool(processes=10) as pool:     
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, pool=pool)
        sampler.run_mcmc(p0, nsteps, progress=True)

    dis = 1000
    thi = 30

    chains = sampler.get_chain(flat=True, discard=dis, thin=thi)

    name = ['Omega0' ,'Sigma0', 'H0', 'M', 'rd']
    labels1 = [r'\Omega_0', r'\sigma_0', r'H_0', r'\mathcal{M}', r'r_d']

    sample2 = MCSamples(samples=chains, names=name, labels=labels1)

    np.savetxt("chains_1.txt", chains, delimiter="\t")

    #  Now properly recompute the pure likelihood values ---
    print("Now computing the pure log-likelihoods...")

    likelihoods_only = []
    samples = sampler.get_chain(flat=True, discard=dis, thin=thi)

    for param_set in tqdm(samples):
        _, likelihood = log_posterior(param_set, return_likelihood_only=True)
        likelihoods_only.append(likelihood)

    likelihoods_only = np.array(likelihoods_only)

    max_log_likelihood = np.max(likelihoods_only)
    
    N = 1618
    k = ndim  # number of parameters

    chi2_values = -2 * likelihoods_only
    min_chi2 = np.min(chi2_values)
    reduced_chi2 = min_chi2 / (N - k)
    p_value = 1 - stats.chi2.cdf(min_chi2, df=(N - k))

    AIC = -2 * max_log_likelihood + 2 * k
    BIC = -2 * max_log_likelihood + k * np.log(N)

    print("Minimum chi-squared =", min_chi2)
    print("Reduced chi-squared =", reduced_chi2)
    print("P-value =", p_value)
    print("Maximum log-likelihood =", max_log_likelihood)
    print("AIC =", AIC)
    print("BIC =", BIC)
	
    g = plots.get_subplot_plotter(width_inch=10)
    g.settings.figure_legend_frame = True
    g.settings.alpha_filled_add = 0.6
    g.settings.title_limit_fontsize = 9.5
    g.settings.axes_labelsize = 14
    g.settings.legend_fontsize = 16
    g.settings.colorbar_axes_fontsize = 10

    g.triangle_plot(sample2, ['Omega0','Sigma0', 'H0', 'rd'], filled=True, legend_labels=['Linear Model'], legend_loc='upper right', contour_colors=['darkblue'], title_limit=1)

    g.export('fig_1.pdf')

if __name__ == '__main__':
    main()
