#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import numpy as np
import matplotlib.pyplot as plt
import h5py
import pyunlocbox


def nonzero(s):
    """
    Return an array where 1 indicates a non-zero element and 0 a zero element.
    """
    ind = np.abs(s) != 0
    N = np.sum(ind)
    return ind, N


def plotfftreal(s, fs, title, xlim=None, amp='abs'):
    """
    Plot the Fourier transform in a nice way.

    Parameters
    ----------
    s : array_like
        Fourier transform of a signal.
    fs : float
        Sampling frequency.
    title : string
        Title of the graph.
    xlim : tuple
        X-axis limits.
    amp : {'abs', 'real', 'imag'}
        Type of amplitude to plot.
    """
    N = len(s)
    w = np.linspace(0, fs-fs/N, N)

    exec('y = np.%s(s)' % (amp,))

    plt.plot(w, y, 'b.-')

    plt.grid(True)

    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude (%s)' % (amp,))

    _, N = nonzero(s)
    plt.title('%s (%d)' % (title, N))

    # Force axis numbers to be printed in scientific notation
    plt.ticklabel_format(style='sci', scilimits=(3,3), axis='both')

    if xlim:
        plt.xlim(xlim)


def plot(sf, ylf, yhf, sol1, sol2, fs, xlim=None, filename=None):

    # Set figure size when saving
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
#        plt.savefig(filename + '.pdf')


def artificial():
    """
    Artificial signal composed by a sum of sine waves.
    """
    Ns = 5                  # Number of sines
    Amin = 1                # Minimum/Maximum amplitude for the sine
    Amax = 2
    fs = 1000               # Sampling frequency
    T = 5                   # Sampling / Measurement time
    Ttot = 100              # Total time
    sigma = 1.0             # Noise level

    Nmes = int(fs * T)      # Number of measured samples
    Ntot = int(fs * Ttot)   # Total number of samples

    # We want our estimation to be close to the measures up to the noise level.
    # y = x + epsilon  -->  || Ax - y ||_2 <= || epsilon ||_2
    # Var(eps) = E(eps^2) - E(eps)^2
    # E( ||eps||_2 ) = sqrt( E( ||eps||_2^2 )) = sqrt( sum( E( eps_i^2 ))) = sqrt( N*Var(eps))
    # 1.1 is meant to leave some room.
    epsilon = 1.1 * np.sqrt(Nmes) * sigma  # Radius of the B2-ball

    s = np.zeros(Ntot)

    # Create the sinusoids
    for k in range(Ns):
        f = np.round(np.random.uniform()*Ntot) / Ntot
        amp = Amin + np.random.uniform() * (Amax-Amin)
        s += amp * np.sin( 2 * np.pi * f * np.arange(Ntot))

    # Add noise
    sn = s + sigma * np.random.normal()

    # No offset, otherwise it can create an additional dirac at 0
    sn -= np.mean(sn)

    return sn, fs, Ntot, Nmes, epsilon


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

    # Open Hierarchical Data Format (HDF)
    f = h5py.File(filename)

    # Show datasets or groups
    #f.values()

    # Get signal
    s = f.get('signal')

    # Get sampling frequency
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
    # Signal and sampling frequency
    s, fs = signal('2-calmix.hdf5')

    # Percentage of measured data
    Pmes = 0.05
    Ntot = len(s)
    Nmes = int(Ntot * Pmes)

    # Radius of the B2-ball
    epsilon = 0

    return s, fs, Ntot, Nmes, epsilon


def myoglobin():
    """
    A low resolution signal. Our task is to improve its resolution in the
    Fourier domain and identify the diracs composing the signal. We know that
    it is sparse in the Fourier domain. We have no ground truth.
    """
    # Signal and sampling frequency
    s, fs = signal('1-myoglobin_simplified.hdf5')

    # Percentage of measured data
    Pmes = 0.05
    Nmes = len(s)
    Ntot = int(Nmes / Pmes)

    # Radius of the B2-ball
    epsilon = 0
#    epsilon = 1.1 * np.sqrt(Ntot) * 0.001

    return s, fs, Ntot, Nmes, epsilon


###  Main script  ###


if pyunlocbox.__version__ < '0.2.1':
    raise Exception('PyUNLocBox package older than 0.2.1 contains a bug that '
                    'prevent the correct execution of this script. Current '
                    'version is %s' % (pyunlocbox.__version__,))


###  Parameters  ###


dataset       = 'myoglobin'      # Dataset: artificial, calmix or myoglobin
maxit1        = 50            # Maximum number of iterations for sparse coding
maxit2        = 30            #                                  regression
tol           = 10e-10        # Tolerance to stop iterating
prior_weight  = 1             # Weight of the prior term. Data fidelity has 1.
save_results  = True          # Save or interactively show the results


###  Signal creation  ###


print('Dataset : %s' % (dataset,))
exec('s, fs, Ntot, Nmes, epsilon = %s()' % (dataset,))
print('%d measures out of %d samples (%d%%)' % (Nmes, Ntot, 100.*Nmes/Ntot))

# Masking matrix
mask = np.zeros(Ntot)
mask[:Nmes] = 1

# Low resolution measurements : s or part of s if s is the ground truth
yl = s[:Nmes]
ylf = np.fft.rfft(yl)

# High resolution measurements : yl followed by zeros
# This is to increase the frequency resolution in Fourier
yh = np.zeros(Ntot)
yh[:Nmes] = yl
yhf = np.fft.rfft(yh)

# Ground truth if any
if len(s) == Ntot:
    sf = np.fft.rfft(s)
else:
    sf = None


###  Problem 1 : Sparse coding  ###


# Data fidelity term
A = lambda x: mask * np.fft.irfft(x)
At = lambda x: np.fft.rfft(mask * x)
f1 = pyunlocbox.functions.proj_b2(A=A, At=At, y=yh, nu=1, tight=True,
                                  epsilon=epsilon)

# Prior term
f2 = pyunlocbox.functions.norm_l1(lambda_=prior_weight)

# Solver : Douglas-Rachford as we have no gradient
solver = pyunlocbox.solvers.douglas_rachford(step=np.max(np.abs(yhf)))

# Solve the problem
x0 = np.zeros(np.shape(yhf))
sol1 = pyunlocbox.solvers.solve([f1, f2], x0, solver, rtol=tol, maxit=maxit1,
                                verbosity='LOW')

# Non-zero values indicate peaks
ind1, N = nonzero(sol1['sol'])
print('Number of non-zero coefficients (step 1) : %d' % (N,))


###  Problem 2 : Regroup aggregates into diracs  ###


# As the solution is sparse, the diracs (actually aggregates) are separated by
# zeros. We group together individual chunks of non-zero bins.

tstart_step2 = time.time()

# Transitions from non-aggregate to aggregate
# ind[0:-1] 0 0 1 1 1 1 0 0
# ind[1:]   0 1 1 1 1 0 0 0
# trans     0 1 0 0 0 1 0 0
trans = ind1[0:-1] != ind1[1:]

# Indices of transitions : starts and ends
nz = np.nonzero(trans)[0]
starts = nz[0::2]
ends = nz[1::2]

Npeaks = np.sum(trans) / 2.

ind2 = np.zeros(np.shape(ind1))

# Find the maximum of each aggregate, where the dirac most probably sits
for k in range(int(Npeaks)):
    idx = np.argmax(np.abs(sol1['sol'][starts[k]:ends[k]]))
    idx += starts[k]
    ind2[idx] = 1

if not len(starts) == len(ends) == np.sum(ind2) == Npeaks:
    raise Exception('Aggregates grouping failed')

print('Number of non-zero coefficients (step 2) : %d' % (Npeaks,))

time_step2 = time.time() - tstart_step2


###  Problem 3 : Estimate dirac amplitudes through linear regression  ###


# Now that we have the indices of the diracs, we can force the other
# coefficients to zero and minimize the L2-norm by gradient descent with
# forward-backward. This is a linear regression problem with a constraint.

# New problem : argmin_x || M F^-1 x - y ||_2^2  s.t.  x[ind] = 0

# This tries to approach the measurements with the allowed frequency bins.
# It'll only be useful if the bins are the right ones.
# If there is too much of them, it'll simply retrieve the measurements.

# Gradient descent under constraint with forward-backward
solver = pyunlocbox.solvers.forward_backward()

# Data fidelity term is the same than before, but expressed as a function
# to minimize instead of a constraint
f1 = pyunlocbox.functions.norm_l2(A=A, At=At, y=yh)

# The prior is the indices who can be different than 0. The proximal
# operator is a projection on the constraint set.
f2 = pyunlocbox.functions.func()
f2._eval = lambda x: 0
f2._prox = lambda x, T: ind2 * x

# Start from zero or last solution
x0 = np.zeros(np.shape(yhf))
#x0 = sol1['sol']

# Solve the problem
sol2 = pyunlocbox.solvers.solve([f1, f2], x0, solver, rtol=tol,
                                maxit=maxit2, verbosity='LOW')

# Non-zero values indicate peaks
ind2, N = nonzero(sol2['sol'])
print('Number of non-zero coefficients (step 3) : %d' % (N,))


###  Show results  ###

# Verify the exactitude of the algorithm on the artificial dataset
if dataset is 'artificial':
    inds = np.abs(sf) >= np.max(np.abs(sf)/2)
    if not np.array_equal(inds,ind2):
        print('Number of errors : %d' % (np.sum(inds != ind2),))
        print('    Ground truth : %s' % (str(np.nonzero(inds)[0]),))
        print('    Solution : %s' % (str(np.nonzero(ind2)[0]),))

# Time measurements
print('Elapsed time :')
print('    Step 1 : %.2f seconds' % (sol1['time'],))
print('    Step 2 : %.2f seconds' % (time_step2,))
print('    Step 3 : %.2f seconds' % (sol2['time'],))
print('    Total : %.2f seconds' % (sol1['time'] + time_step2 + sol2['time'],))

# Full view
filename = dataset+'_full' if save_results else None
plot(sf, ylf, yhf, sol1, sol2, fs, None, filename)

# Partially zoomed view
if dataset is 'calmix':
    xlim = (160e3, 300e3)
elif dataset is 'myoglobin':
    xlim = (938.5e3, 941.5e3)
elif dataset is 'artificial':
    # Show the third dirac
    width = 80
    dirac3 = np.nonzero(ind2)[0][2]
    xlim = np.array([dirac3 - width, dirac3 + width], dtype=float)
    xlim *= float(fs) / len(ind2)
    xlim = tuple(xlim)
else:
    xlim = None
if xlim:
    filename = dataset+'_zoom1' if save_results else None
    plot(sf, ylf, yhf, sol1, sol2, fs, xlim, filename)

# Completely zoomed view
if dataset is 'calmix':
    xlim = (245.05e3, 245.55e3)
elif dataset is 'myoglobin':
    xlim = (940.1e3, 940.45e3)
else:
    xlim = None
if xlim:
    filename = dataset+'_zoom2' if save_results else None
    plot(sf, ylf, yhf, sol1, sol2, fs, xlim, filename)

# Interactively show results if not saved to figures
if not save_results:
    plt.show()
