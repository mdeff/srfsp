import numpy as np
import matplotlib.pyplot as plt
import h5py
import pyunlocbox


def plotfftreal(s, fs, xlim=None, amp='abs'):
    r"""
    Plot the Fourier transform in a nice way.

    Parameters
    ----------
    s : array_like
        Fourier transform of a signal.
    fs : float
        Sampling frequency.
    xlim : tuple
        X-axis limits.
    amp : {'abs', 'real', 'imag'}
        Type of amplitude to plot.
    """
    N = len(s)
    w = np.linspace(0, fs-fs/N, N)

    exec('y = np.%s(s)' % (amp,))

    plt.plot(w, y, 'b.-')

    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude (%s)' % (amp,))

    # Force axis numbers to be printed in scientific notation
    plt.ticklabel_format(style='sci', scilimits=(3,3), axis='both')

    if xlim:
        plt.xlim(xlim)


def plot(sf, yf, sol1, sol2, fs, xlim=None, filename=None):

    plt.figure(figsize=(20,10), dpi=200)
#    plt.figure()

    plt.subplot(2,3,1)
    plotfftreal(sf, fs, xlim)
    plt.title('Ground truth')

    plt.subplot(2,3,2)
    plotfftreal(yf, fs, xlim)
    plt.title('Measurements')

    plt.subplot(2,3,4)
    plotfftreal(sol1['sol'], fs, xlim)
    N = np.sum(np.abs(sol1['sol']) != 0)
    plt.title('Recovered after sparsity constraint (%d)' % (N,))

    if sol2:

        plt.subplot(2,3,5)
        plotfftreal(sol2['sol'], fs, xlim)
        N = np.sum(np.abs(sol2['sol']) != 0)
        plt.title('Recovered after linear regression (%d)' % (N,))

        plt.subplot(2,3,6)
        plt.plot(sol2['objective'])
        plt.title('Objective function')
        plt.ticklabel_format(style='sci', scilimits=(3,3), axis='both')

    else:

        plt.subplot(2,3,5)
        plotfftreal(sol1['sol'], fs, xlim, amp='real')
        plt.title('Real part')

        plt.subplot(2,3,6)
        plotfftreal(sol1['sol'], fs, xlim, amp='imag')
        plt.title('Imaginary part')

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

    Nmes = fs * T           # Number of measured samples
    Ntot = fs * Ttot        # Total number of samples

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

    return sn, fs, Ntot, Nmes, epsilon


def signal(filename, Pmes):
    """
    Import a signal from an HDF file.
    """

    # Open Hierarchical Data Format (HDF)
    f = h5py.File(filename)

    # Show datasets or groups
    #f.values()

    # Get signal
    s = f.get('signal')

    # Get sampling frequency
    fs = f.get('fs').value

    Ntot = len(s)
    Nmes = np.round(Pmes * Ntot)

    # Radius of the B2-ball
    epsilon = 0
#    epsilon = 1.1 * np.sqrt(Ntot) * 0.001

    return s, fs, Ntot, Nmes, epsilon


def calmix():
    Pmes = 0.05  # Percentage of measured data
    return signal('2-calmix.hdf5', Pmes)


def myoglobin():
    Pmes = 0.50  # Percentage of measured data
    return signal('1-myoglobin_simplified.hdf5', Pmes)


###  Parameters  ###


dataset       = 'calmix'      # Dataset: artificial, calmix or myoglobin
maxit1        = 50            # Maximum number of iterations for sparse coding
maxit2        = 15            #                                  regression
tol           = 10e-10        # Tolerance to stop iterating
do_regression = True          # Do a linear regression as a third step
prior_weight  = 1             # Weight of the prior term. Data fidelity has 1.


###  Signal creation  ###


exec('s, fs, Ntot, Nmes, epsilon = %s()' % (dataset,))

# Masking matrix
mask = np.zeros(Ntot)
mask[0:Nmes] = 1

# Measurements
y = mask * s
yf = np.fft.rfft(y)

# Ground truth
sf = np.fft.rfft(s)


###  Problem 1 : Sparse coding  ###


# Data fidelity term
A = lambda x: mask * np.fft.irfft(x)
At = lambda x: np.fft.rfft(mask * x)
f1 = pyunlocbox.functions.proj_b2(A=A, At=At, y=y, nu=1, tight=True,
                                  epsilon=epsilon)

# Prior term
f2 = pyunlocbox.functions.norm_l1(lambda_=prior_weight)

# Solver : Douglas-Rachford as we have no gradient
solver = pyunlocbox.solvers.douglas_rachford(step=np.max(np.abs(yf)))

# Solve the problem
x0 = np.zeros(np.shape(yf))
sol1 = pyunlocbox.solvers.solve([f1, f2], x0, solver, rtol=tol, maxit=maxit1,
                                verbosity='LOW')

# Non-zero values indicate peaks
ind1 = np.abs(sol1['sol']) != 0
print('Number of non-zero coefficients : %d' % (np.sum(ind1),))


###  Problem 2 : Regroup aggregates into diracs  ###


# As the solution is sparse, the diracs (actually aggregates) are separated by
# zeros. We group together individual chunks of non-zero bins.

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
    idx = np.argmax(sol1['sol'][starts[k]:ends[k]])
    idx += starts[k]
    ind2[idx] = 1

if not len(starts) == len(ends) == np.sum(ind2) == Npeaks:
    raise Exception('Aggregates grouping failed')

print('Number of non-zero coefficients : %d' % (Npeaks,))


###  Problem 3 : Estimate dirac amplitudes through linear regression  ###


# Now that we have the indices of the diracs, we can force the other
# coefficients to zero and minimize the L2-norm by gradient descent with
# forward-backward. This is a linear regression problem with a constraint.

# New problem : argmin_x || M F^-1 x - y ||_2^2  s.t.  x[ind] = 0

# This tries to approach the measurements with the allowed frequency bins.
# It'll only be useful if the bins are the right ones.
# If there is too much of them, it'll simply retrieve the measurements.

if do_regression:

    # Gradient descent under constraint with forward-backward
    solver = pyunlocbox.solvers.forward_backward()

    # Data fidelity term is the same than before, but expressed as a function
    # to minimize instead of a constraint
    f1 = pyunlocbox.functions.norm_l2(A=A, At=At, y=y)

    # The prior is the indices who can be different than 0. The proximal
    # operator is a projection on the constraint set.
    f2 = pyunlocbox.functions.func()
    f2._eval = lambda x: 0
    f2._prox = lambda x, T: ind2 * x

    # Start from zero or last solution
    x0 = np.zeros(np.shape(yf))
    #x0 = sol1['sol']

    # Solve the problem
    sol2 = pyunlocbox.solvers.solve([f1, f2], x0, solver, rtol=tol,
                                    maxit=maxit2, verbosity='LOW')

    # Non-zero values indicate peaks
    ind3 = np.abs(sol2['sol']) != 0
    print('Number of non-zero coefficients : %d' % (np.sum(ind3),))

else:

    sol2 = None


###  Show results  ###


# Full view
plot(sf, yf, sol1, sol2, fs, filename=dataset+'_full')

# Partially zoomed view
if dataset is 'calmix':
    xlim = (160e3, 300e3)
elif dataset is 'myoglobin':
    xlim = (938.5e3, 941.5e3)
else:
    xlim = None
if xlim:
    plot(sf, yf, sol1, sol2, fs, xlim, dataset+'_zoom1')

# Completely zoomed view
if dataset is 'calmix':
    xlim = (245.05e3, 245.55e3)
else:
    xlim = None
if xlim:
    plot(sf, yf, sol1, sol2, fs, xlim, dataset+'_zoom2')

#plt.show()
