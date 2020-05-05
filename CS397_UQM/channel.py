#!/bin/py
import numpy as np
import matplotlib
matplotlib.use('tkagg')
from matplotlib import pyplot
import pylab
import matplotlib.mlab as mlab
import matplotlib.ticker as ticker
from scipy import stats
import sys
import h5py

# local files that will be imported
import prior
import likelihood

# construct map of prior functions, to plot below
fdict = {'prior_q': prior.prior_q,
         'prior_beta': prior.prior_beta,
         'prior_k': prior.prior_k,
         'prior_c1': prior.prior_c1,
         'prior_c2': prior.prior_c2,
         'prior_c3': prior.prior_c3,
         'prior_deq': prior.prior_deq,
         'prior_deqq': prior.prior_deqq,
         'prior_diq': prior.prior_diq,
         'prior_delta': prior.prior_delta,
         'prior_gamma': prior.prior_gamma,
         'prior_E0': prior.prior_E0,
         'prior_I0': prior.prior_I0}

# -------------------------------------------------------------
# subroutine that generates a .pdf file plotting a quantity
# -------------------------------------------------------------
def plotter(chain,quant,xmin=None,xmax=None):
    from math import log, pi
    bins = np.linspace(np.min(chain), np.max(chain), 200)
    qkde = stats.gaussian_kde(chain)
    qpdf = qkde.evaluate(bins)
    qpdf = qpdf/np.linalg.norm(qpdf) 
    # plot posterior
    pyplot.figure()
    pyplot.plot(bins, qpdf, linewidth=3, label="Post")

    # plot prior (requires some cleverness to do in general)
    qpr  = [fdict['prior_'+quant](x) for x in bins]
    qpri = [np.exp(x) for x in qpr]        
    qpri=qpri/np.linalg.norm(qpri) 
    pyplot.plot(bins, qpri, linewidth=3, label="Prior")

    # user specified bounds to x-range:
    if(xmin != None and xmax != None):
        bounds = np.array([xmin, xmax])
        pyplot.xlim(bounds)        
    
    pyplot.xlabel(quant, fontsize=30)
    pyplot.ylabel('$\pi('+quant+')$', fontsize=30)
    pyplot.legend(loc='upper left')
    pyplot.savefig(quant+'_post.png', bbox_inches='tight')
    

# -------------------------------------------------------------
# MCMC sampling Function
# -------------------------------------------------------------

class BayesianRichardsonExtrapolation(object):
    "Computes the Bayesian Richardson extrapolation posterior log density."

    def __call__(self, params, dtype=np.double):
        q,beta,k,c1,c2,c3,deq,deqq,diq,delta,gamma,E0,I0 = params

        from math import log

        return (
            prior.prior(q,beta,k,c1,c2,c3,deq,deqq,diq,delta,gamma,E0,I0) + 
            likelihood.likelihood(q,beta,k,c1,c2,c3,deq,deqq,diq,delta,gamma,E0,I0)
            )

# -------------------------------------------------------------
# Main Function
# -------------------------------------------------------------
#
# Stop module loading when imported.  Otherwise continue running.
#if __name__ != '__main__': raise SystemExit, 0

# Example of sampling Bayesian Richardson extrapolation density using emcee
from emcee import EnsembleSampler
from math import ceil, floor, sqrt

#
# initalize the Bayesian Calibration Procedure 
#
bre = BayesianRichardsonExtrapolation()

print("\nInitializing walkers")
nwalk = 100

# initial guesses for the walkers starting locations
guess = [0.2522,0.0964,0.1997,0.0101,16.8852,0.1311,0.1776,0.01,0.1194,0.0019,0.0135,5000.,1000.]

params0 = np.tile(guess, nwalk).reshape(nwalk,13)
params0[:,:11] += np.random.rand(nwalk,11) * 0.01 # Perturb Model Parameters
params0.T[11] += np.random.rand(nwalk) * 500 # Perturb E0
params0.T[12] += np.random.rand(nwalk) * 100 # Perturb I0
params0 = np.absolute(params0)       # ...and force >= 0

# Set up the backend
# Don't forget to clear it in case the file already exists
filename = "backend.h5"
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim)


print("\nInitializing the sampler and burning in walkers")
s = EnsembleSampler(nwalk, params0.shape[-1], bre, backend=backend)
#pos, prob, state = s.run_mcmc(params0, 10000, progress=True)
#s.reset()
#print("\nSampling the posterior density for the problem")
#s.run_mcmc(pos, 20000, progress=True)
#print("Mean acceptance fraction: {0:.3f}".format(np.mean(s.acceptance_fraction)))
#print("Mean autocorrelation time: {0:.3f} steps".format(np.mean(s.get_autocorr_time())))

#
#convergence-based MCMC
#
max_n = 100000

# We'll track how the average autocorrelation time estimate changes
index = 0
autocorr = np.empty(max_n)

# This will be useful to testing convergence
old_tau = np.inf

# Now we'll sample for up to max_n steps
for sample in sampler.sample(params0, iterations=max_n, progress=True):
    # Only check convergence every 100 steps
    if sampler.iteration % 100:
        continue

    # Compute the autocorrelation time so far
    # Using tol=0 means that we'll always get an estimate even
    # if it isn't trustworthy
    tau = sampler.get_autocorr_time(tol=0)
    autocorr[index] = np.mean(tau)
    index += 1

    # Check convergence
    converged = np.all(tau * 100 < sampler.iteration)
    converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
    if converged:
        break
    old_tau = tau
#
# graphical convergence check
#
n = 100 * np.arange(1, index + 1)
y = autocorr[:index]
pyplot.plot(n, n / 100.0, "--k")
pyplot.plot(n, y)
pyplot.xlim(0, n.max())
pyplot.ylim(0, y.max() + 0.1 * (y.max() - y.min()))
pyplot.xlabel("number of steps")
pyplot.ylabel(r"mean $\hat{\tau}$")
pyplot.savefig('convergence.png', bbox_inches='tight')

#
# 1d Marginals
#
print("\nDetails for posterior one-dimensional marginals:")
def textual_boxplot(label, unordered, header):
    n, d = np.size(unordered), np.sort(unordered)
    if (header): print((10*" %15s") % ("", "min", "P5", "P25", "P50", "P75", "P95", "max", "mean", "stddev"))
    print((" %15s" + 9*" %+.8e") % (label,
                                    d[0],
                                    d[[floor(1.*n/20), ceil(1.*n/20)]].mean(),
                                    d[[floor(1.*n/4), ceil(1.*n/4)]].mean(),
                                    d[[floor(2.*n/4), ceil(2.*n/4)]].mean(),
                                    d[[floor(3.*n/4), ceil(3.*n/4)]].mean(),
                                    d[[floor(19.*n/20), ceil(19.*n/20)]].mean(),
                                    d[-1],
                                    d.mean(),
                                    d.std()))
    #return d[[floor(1.*n/20), ceil(1.*n/20)]].mean(), d[[floor(17.*n/20), ceil(17.*n/20)]].mean()
    return d.mean(), 2*d.std()

qm, qs = textual_boxplot("q", s.flatchain[:,0], header=True)
betam, betas = textual_boxplot("beta", s.flatchain[:,1], header=False)
km, ks = textual_boxplot("k", s.flatchain[:,2], header=False)
c1m, c1s = textual_boxplot("c1", s.flatchain[:,3], header=False)
c2m, c2s = textual_boxplot("c2", s.flatchain[:,4], header=False)
c3m, c3s = textual_boxplot("c3", s.flatchain[:,5], header=False)
deqm, deqs = textual_boxplot("deq", s.flatchain[:,6], header=False)
deqqm, deqqs = textual_boxplot("deqq", s.flatchain[:,7], header=False)
diqm, diqs = textual_boxplot("diq", s.flatchain[:,8], header=False)
deltam, deltas = textual_boxplot("delta", s.flatchain[:,9], header=False)
gammam, gammas = textual_boxplot("gamma", s.flatchain[:,10], header=False)
E0m, E0s = textual_boxplot("E0", s.flatchain[:,11], header=False)
I0m, I0s = textual_boxplot("I0", s.flatchain[:,12], header=False)

#----------------------------------
# FIGURES: Marginal posterior(s)
#----------------------------------
print("\nPrinting PDF output")

plotter(s.flatchain[:,0],'q')
plotter(s.flatchain[:,1],'beta')
plotter(s.flatchain[:,2],'k')
plotter(s.flatchain[:,3],'c1')
plotter(s.flatchain[:,4],'c2')
plotter(s.flatchain[:,5],'c3')
plotter(s.flatchain[:,6],'deq')
plotter(s.flatchain[:,7],'deqq')
plotter(s.flatchain[:,8],'diq')
plotter(s.flatchain[:,9],'delta')
plotter(s.flatchain[:,10],'gamma')
plotter(s.flatchain[:,11],'E0')
plotter(s.flatchain[:,12],'I0')

