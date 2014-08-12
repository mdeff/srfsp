import numpy as np
import matplotlib.pyplot as plt
import pyunlocbox
from tools import plotfftreal


###  Parameters  ###


Ns = 5                  # Number of sines
Amin = 1                # Minimum/Maximum amplitude for the sine
Amax = 2
fs = 1000               # Sampling frequency
T = 5                   # Sampling / Measurement time
Ttot = 100              # Total time
sigma = 1.0             # Noise level
do_regression = False   # Do a linear regression as a second step

Nmes = fs * T
Ntot = fs * Ttot

# We want our estimation to be close to the measures up to the noise level.
# y = x + epsilon  -->  || Ax - y ||_2 <= || epsilon ||_2
# Var(eps) = E(eps^2) - E(eps)^2
# E( ||eps||_2 ) = sqrt( E( ||eps||_2^2 )) = sqrt( sum( E( eps_i^2 ))) = sqrt( N*Var(eps))
# 1.1 is meant to leave some room.
epsilon = 1.1 * np.sqrt(Nmes) * sigma  # Radius of the B2-ball

maxit     = 50        # Maximum number of iteration
tol       = 10e-10      # Tolerance to stop iterating

#show_evolution = True  # show evolution of the algorithm


###  Signal creation  ###


s = np.zeros(Ntot)

# Create the sinusoids
for k in range(Ns):
    f = np.round(np.random.uniform()*Ntot) / Ntot
    amp = Amin + np.random.uniform() * (Amax-Amin)
    s += amp * np.sin( 2 * np.pi * f * np.arange(Ntot))

# Add noise
sn = s + sigma * np.random.normal()

# Masking matrix
mask = np.zeros(Ntot)
mask[0:Nmes] = 1

# Measurements
y = mask * sn
yf = np.fft.rfft(y)

# Ground truth
sf = np.fft.rfft(s)


###  Problem 1 : find peaks  ###


# Data fidelity term
A = lambda x: mask * np.fft.irfft(x)
At = lambda x: np.fft.rfft(mask * x)
f1 = pyunlocbox.functions.proj_b2(A=A, At=At, y=y, nu=1, tight=True,
                                  epsilon=epsilon)

# Prior term
f2 = pyunlocbox.functions.norm_l1()

# Solver : Douglas-Rachford as we have no gradient
solver = pyunlocbox.solvers.douglas_rachford(step=np.max(np.abs(yf)))

# Solve the problem
x0 = np.zeros(np.shape(yf))
ret = pyunlocbox.solvers.solve([f1, f2], x0, solver, rtol=tol, maxit=maxit, verbosity='LOW')

sol1 = ret['sol']

# Non-zero terms --> indices of the diracs
ind = np.abs(sol1) > 1.
print('Number of non-zero coefficients : %d' % (np.sum(ind),))


###  Problem 2 : find amplitudes  ###


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
    f2._prox = lambda x, T: ind * x

    # Start from zero or last solution
    x0 = np.zeros(np.shape(yf))
    #x0 = sol1

    # Solve the problem
    ret = pyunlocbox.solvers.solve([f1, f2], x0, solver, rtol=tol, maxit=20,
                                   verbosity='LOW')
    sol2 = ret['sol']

    # Non-zero terms --> indices of the diracs
    ind = np.abs(sol2)
    print('Number of non-zero coefficients : %d' % (np.sum(ind > 1.),))


###  Results  ###


plt.figure()

plt.subplot(2,3,1)
plotfftreal(sf, fs)
plt.title('Ground truth')

plt.subplot(2,3,2)
plotfftreal(np.fft.rfft(mask*s), fs)
plt.title('Measurements without noise')

plt.subplot(2,3,3)
plotfftreal(yf, fs)
plt.title('Measurements with noise')

plt.subplot(2,3,4)
plotfftreal(sol1, fs)
plt.title('Recovered after sparsity constraint')

if do_regression:

    plt.subplot(2,3,5)
    plotfftreal(sol2, fs)
    plt.title('Recovered after linear regression')

    plt.subplot(2,3,6)
    plt.plot(ret['objective'])
    plt.title('Objective function')

else:

    plt.subplot(2,3,5)
    plotfftreal(sol1, fs, amp='real')
    plt.title('Real part')

    plt.subplot(2,3,6)
    plotfftreal(sol1, fs, amp='imag')
    plt.title('Imaginary part')

plt.show()
