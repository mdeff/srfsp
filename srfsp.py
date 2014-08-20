#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import numpy as np
import matplotlib.pyplot as plt
import h5py
import pyunlocbox


def nonzero(s):
    """
    Return an array where ``True`` indicates a non-zero element and ``False`` a
    zero element.

    Parameters:
    -----------
    s : array_like
        Input signal.

    Returns
    -------
    ind : ndarray of bool
        ind[k] == s[k] != 0
    N : int
        Number of non-zero elements.

    Examples
    --------
    >>> ind, N = nonzero([0,1,2,3,0])
    >>> ind
    array([False,  True,  True,  True, False], dtype=bool)
    >>> N
    3
    """
    ind = np.abs(s) != 0
    N = np.sum(ind)
    return ind, N


def plotfftreal(sf, fs, title, xlim=None, amp='abs'):
    """
    Plot the Fourier transform in a nice way.

    Parameters
    ----------
    sf : array_like
        Fourier transform of a signal.
    fs : float
        Sampling frequency.
    title : string
        Title of the graph.
    xlim : tuple of floats, optional
        X-axis limits. Default is automatic.
    amp : {'abs', 'real', 'imag'}, optional
        Type of amplitude to plot. Default is abs.
    """
    N = len(sf)
    w = np.linspace(0, fs-fs/N, N)

    exec('y = np.%s(sf)' % (amp,))

    plt.plot(w, y, 'b.-')

    plt.grid(True)

    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude (%s)' % (amp,))

    _, N = nonzero(s)
    plt.title('%s (%d)' % (title, N))

    # Force axis numbers to be printed in scientific notation.
    plt.ticklabel_format(style='sci', scilimits=(3,3), axis='both')

    if xlim:
        plt.xlim(xlim)


def plotresults(sf, ylf, yhf, sol1, sol2, fs, xlim=None, filename=None):
    """
    Plot the results.

    Parameters
    ----------
    sf : array_like
        Ground truth in Fourier.
    ylf : array_like
        Low resolution measurements.
    yhf : array_like
        High resolution measurements.
    sol1 : array_like
        Solution after the first optimization step.
    sol2 : array_like
        Solution after the second optimization step.
    fs : float
        Sampling frequency.
    xlim : tuple of floats, optional
        X-axis limits. Default is automatic.
    filename : string, optional
        Name of the saved figure file in the current directory or path to it.
        Nothing is saved if None (default).
    """
    # Set figure size when saving.
    if filename:
        plt.figure(figsize=(20,10), dpi=200)
    else:
        plt.figure()

    if sf is not None:
        plt.subplot(2,3,1)
        title = 'Ground truth'
        plotfftreal(sf, fs, title, xlim)

    plt.subplot(2,3,2)
    title = 'Measurements (low res)'
    plotfftreal(ylf, fs, title, xlim)

    plt.subplot(2,3,3)
    title = 'Measurements (high res)'
    plotfftreal(yhf, fs, title, xlim)

    plt.subplot(2,3,4)
    title = 'Recovered 1 (sparsity constraint)'
    plotfftreal(sol1['sol'], fs, title, xlim)

    plt.subplot(2,3,5)
    title = 'Recovered 2 (linear regression)'
    plotfftreal(sol2['sol'], fs, title, xlim)

    plt.subplot(2,3,6)
    plt.semilogy(sol2['objective'], 'b.-')
    plt.grid(True)
    plt.title('Objective function (linear regression)')
    plt.xlabel('Iteration number')

    if filename:
        plt.savefig(filename + '.png')
        #plt.savefig(filename + '.pdf')


def addnoise(s, Nmes, sigma=1.0):
    """
    Add some white noise to a signal.

    Parameters
    ----------
    s : array_like
        Noiseless signal.
    Nmes : int
        Number of measured samples. Used to compute epsilon, as the algorithm
        only 'sees' the noise on the measurements.
    sigma : float, optional
        Noise level. Default is 1.

    Returns
    -------
    s : ndarray
        Noisy signal.
    epsilon : float
        Radius of the L2-ball.

    Notes
    -----
    The radius of the L2-ball, `epsilon`, is a measure of the confidence we
    have in our measurements. Smaller is the radius, closer to the measurements
    we will be. A too small radius can lead to over-fitting. A too big radius
    will cause the algorithm to go away from the measurements.

    The ideal is an estimation that is close to the measures up to the noise
    level. The noisy signal y is composed by the sum of the noiseless signal x
    and the white noise sigma : y = x + sigma. We thus want
    ||Ax-y||_2 <= ||sigma||_2. An estimation of it is :
    E(||sigma||_2) = sqrt(E(||sigma||_2^2)) = sqrt(sum(E(sigma_i^2)))
    As Var(sigma) = E(sigma^2) - E(sigma)^2 and E(sigma)^2 = 0, we have :
    E(||sigma||_2) = sqrt(N * Var(eps))
    The coefficient 1.1 is meant to leave some room, but 1.0 works best for
    high noise levels.
    """
    if sigma != 0:
        sn = s + np.random.normal(0, sigma, len(s))
    else:
        sn = s
    epsilon = 1.0 * np.sqrt(Nmes) * sigma
    return sn, epsilon


def artificial():
    """
    Artificial signal composed by a sum of sine waves.
    """
    Ns = 5                  # Number of sines.
    Amin = 1                # Minimum/Maximum amplitude for the sines.
    Amax = 2
    fs = 1000               # Sampling frequency.
    Tmes = 5                # Measurement time.
    Ttot = 100              # Total time.

    Nmes = int(fs * Tmes)   # Number of measured samples.
    Ntot = int(fs * Ttot)   # Total number of samples.

    # Create the sum of sinusoids.
    s = np.zeros(Ntot)
    #np.random.seed(15)
    for k in range(Ns):
        f = np.round(np.random.uniform() * Ntot) / Ntot
        amp = Amin + np.random.uniform() * (Amax-Amin)
        s += amp * np.sin(2 * np.pi * f * np.arange(Ntot))

    return s, fs, Ntot, Nmes


def signal(filename):
    """
    Import a signal, along with its sampling frequency, from an HDF file.

    Parameters
    ----------
    filename : string
        Name of the HDF file in the current directory or path to it.

    Returns
    -------
    s : array_like
        The retrieved signal.
    fs : float
        The signal sampling frequency.
    """
    # Open Hierarchical Data Format (HDF).
    f = h5py.File(filename)

    # Show datasets or groups.
    #f.values()

    # Get signal and sampling frequency.
    s = f.get('signal')
    fs = f.get('fs').value

    print('Sampling frequency : %d MHz, # samples : %d' % (fs/1e6, len(s)))
    return s, fs


def calmix():
    """
    A high resolution signal, i.e. the ground truth. The measurement is a
    masked version of it (i.e. we retain only a portion of it). We then try to
    recover the information from the measurement and compare it to the ground
    truth. We know that the signal is sparse in the Fourier domain.
    """
    # Signal and sampling frequency.
    s, fs = signal('2-calmix.hdf5')

    # Percentage of measured data.
    Pmes = 0.05
    Ntot = len(s)
    Nmes = int(Ntot * Pmes)

    return s, fs, Ntot, Nmes


def myoglobin():
    """
    A low resolution signal. Our task is to increase its resolution in the
    Fourier domain and identify the diracs composing the signal. We know that
    it is sparse in the Fourier domain. We have no ground truth.
    """
    # Signal and sampling frequency.
    s, fs = signal('1-myoglobin_simplified.hdf5')

    # Percentage of measured data.
    Pmes = 0.25
    Nmes = len(s)
    Ntot = int(Nmes / Pmes)

    return s, fs, Ntot, Nmes


  #####  Main script  #####


if pyunlocbox.__version__ < '0.2.1':
    raise Exception('PyUNLocBox packages older than 0.2.1 contain a bug that '
                    'prevent the correct execution of this script. Current '
                    'version is %s' % (pyunlocbox.__version__,))


  #####  Parameters  #####


dataset       = 'calmix'      # Dataset: artificial, calmix or myoglobin.
maxit1        = 50            # Maximum number of iterations for sparse coding.
maxit2        = 50            #                                  regression.
tol1          = 1e-10         # Tolerance to stop iterating for sparse coding.
tol2          = 1e-3          #                                 regression.
prior_weight  = 1.0           # Weight of the prior term.
sigma         = 1.0           # Noise level.
save_results  = True          # Save or interactively show the results.


  #####  Signal creation  #####


# Get the signal (from a stored dataset or synthesize one).
print('Dataset : %s' % (dataset,))
exec('s, fs, Ntot, Nmes = %s()' % (dataset,))
print('%d measures out of %d samples (%d%%)' % (Nmes, Ntot, 100.*Nmes/Ntot))

# Ground truth if any (before adding noise).
if len(s) == Ntot:
    sf = np.fft.rfft(s)
else:
    sf = None

# Add some white noise (before creating the measurements).
s, epsilon = addnoise(s, Nmes, sigma)

# Low resolution measurements : s or part of s if s is the ground truth.
yl = s[:Nmes]
ylf = np.fft.rfft(yl)

# High resolution measurements : yl followed by zeros.
# This is to increase the frequency resolution in Fourier.
yh = np.zeros(Ntot)
yh[:Nmes] = yl
yhf = np.fft.rfft(yh)


  #####  Step 1 : Sparse coding  #####


# Masking matrix.
mask = np.zeros(Ntot, dtype=bool)
mask[:Nmes] = True

# Data fidelity term : close to measurements.
A = lambda x: mask * np.fft.irfft(x)
At = lambda x: np.fft.rfft(mask * x)
f1 = pyunlocbox.functions.proj_b2(A=A, At=At, y=yh, nu=1, tight=True,
                                  epsilon=epsilon)

# Prior term : sparse in the Fourier domain.
f2 = pyunlocbox.functions.norm_l1(lambda_=prior_weight)

# Solver : Douglas-Rachford as we have no gradient.
solver = pyunlocbox.solvers.douglas_rachford(step=np.max(np.abs(yhf)))

# Solve the problem.
x0 = np.zeros(np.shape(yhf))
sol1 = pyunlocbox.solvers.solve([f1, f2], x0, solver, rtol=tol1, maxit=maxit1,
                                verbosity='LOW')


  #####  Step 2 : Aggregates grouping into diracs  #####


t2start = time.time()

# Non-zero coefficients indicate potential diracs.
ind1, Npot = nonzero(sol1['sol'])

# Transitions from non-aggregate to aggregate (filled with ones).
# ind[:-1] 0 0 0 1 1 1 1 0 0
# ind[1:]  0 0 1 1 1 1 0 0 0
# trans    0 0 1 0 0 0 1 0 0
trans = ind1[:-1] != ind1[1:]

# Indices of transitions : starts and ends.
nz = np.nonzero(trans)[0]
starts = nz[0::2]
ends = nz[1::2]

Ndiracs = np.sum(trans) / 2.

# Find the maximum of each aggregate, where the dirac most probably sits.
ind2 = np.zeros(np.shape(ind1), dtype=bool)
for k in range(int(Ndiracs)):
    idx = np.argmax(np.abs(sol1['sol'][starts[k]:ends[k]]))
    idx += starts[k]
    ind2[idx] = True

if not len(starts) == len(ends) == np.sum(ind2) == Ndiracs:
    raise Exception('Aggregates grouping failed.')

print('Step 2 : %d identified diracs out of %d non-zero coefficients before '
      'aggregates grouping (%d%%)' % (Ndiracs, Npot, 100.*Ndiracs/Npot))

t2 = time.time() - t2start


  #####  Step 3 : Dirac amplitudes estimation through linear regression  #####


# Data fidelity term is the same than before, but expressed as a function
# to minimize instead of a constraint.
f1 = pyunlocbox.functions.norm_l2(A=A, At=At, y=yh)

# The prior (a constraint) is a list of indices that can be non-zero. Its
# proximal operator is a projection on the set that verifies the constraint.
f2 = pyunlocbox.functions.func()
f2._eval = lambda x: 0
f2._prox = lambda x, T: ind2 * x

# Gradient descent under constraint with forward-backward.
solver = pyunlocbox.solvers.forward_backward()

# Start from zero or last solution.
x0 = np.zeros(np.shape(yhf))
#x0 = sol1['sol']

# Solve the problem.
sol2 = pyunlocbox.solvers.solve([f1, f2], x0, solver, rtol=tol2,
                                maxit=maxit2, verbosity='LOW')

# Non-zero coefficients indicate recognized diracs.
ind3, N = nonzero(sol2['sol'])
if N != Ndiracs:
    raise Exception('Constraint on diracs positions not respected.')


###  Results  ###


# Verify the exactitude of the algorithm on the artificial dataset.
if dataset is 'artificial':
    inds = np.abs(sf) >= np.max(np.abs(sf)/2)
    if not np.array_equal(inds, ind3):
        print('Number of errors : %d' % (np.sum(inds != ind3),))
        print('    Ground truth : %s' % (str(np.nonzero(inds)[0]),))
        print('    Solution : %s' % (str(np.nonzero(ind2)[0]),))

# Time measurements.
print('Elapsed time :')
print('    Step 1 : %.2f seconds' % (sol1['time'],))
print('    Step 2 : %.2f seconds' % (t2,))
print('    Step 3 : %.2f seconds' % (sol2['time'],))
print('    Total  : %.2f seconds' % (sol1['time'] + t2 + sol2['time'],))

# Full view.
filename = dataset+'_full' if save_results else None
plotresults(sf, ylf, yhf, sol1, sol2, fs, None, filename)

# Partially zoomed view.
if dataset is 'calmix':
    xlim = (160e3, 300e3)
elif dataset is 'myoglobin':
    xlim = (938.5e3, 941.5e3)
elif dataset is 'artificial':
    # Show the third (or less) dirac if it exists.
    Ndirac = min(N, 2)
    if Ndirac != 0:
        width = 80
        dirac = np.nonzero(ind2)[0][2]
        xlim = np.array([dirac - width, dirac + width], dtype=float)
        xlim *= float(fs) / len(ind2)
        xlim = tuple(xlim)
    else:
        xlim = None
else:
    xlim = None
if xlim:
    filename = dataset+'_zoom1' if save_results else None
    plotresults(sf, ylf, yhf, sol1, sol2, fs, xlim, filename)

# Completely zoomed view.
if dataset is 'calmix':
    xlim = (245.05e3, 245.55e3)
elif dataset is 'myoglobin':
    xlim = (940.1e3, 940.45e3)
else:
    xlim = None
if xlim:
    filename = dataset+'_zoom2' if save_results else None
    plotresults(sf, ylf, yhf, sol1, sol2, fs, xlim, filename)

# Interactively show results if not saved to figures.
if not save_results:
    plt.show()
