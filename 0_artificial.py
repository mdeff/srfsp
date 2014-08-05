import numpy as np
import matplotlib.pyplot as plt
import pyunlocbox
from tools import plotfftreal


###  Parameters  ###


Ns = 5         # Number of sines
Amin = 1       # Minimum/Maximum amplitude for the sine
Amax = 2
fs = 1000      # Sampling frequency
T = 5          # Sampling / Measurement time
Ttot = 100     # Total time
sigma = 1.0    # Noise level

Nmes = fs * T
Ntot = fs * Ttot

# We want our estimation to be close to the measures up to the noise level.
# y = x + epsilon  -->  || Ax - y ||_2 <= || epsilon ||_2
# Var(eps) = E(eps^2) - E(eps)^2
# E( ||eps||_2 ) = sqrt( E( ||eps||_2^2 )) = sqrt( sum( E( eps_i^2 ))) = sqrt( N*Var(eps))
# 1.1 is meant to leave some room.
epsilon = 1.1 * np.sqrt(Nmes) * sigma  # Radius of the B2-ball

maxit     = 100        # Maximum number of iteration
tol       = 10e-10      # Tolerance to stop iterating

show_evolution = True  # show evolution of the algorithm


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


###  Problem setting  ###


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


###  Results  ###


plt.figure()

plt.subplot(2,2,1)
plotfftreal(sf, fs)
plt.title('Ground truth')

plt.subplot(2,2,2)
plotfftreal(np.fft.rfft(mask*s), fs)
plt.title('Measurements without noise')

plt.subplot(2,2,3)
plotfftreal(yf, fs)
plt.title('Measurements with noise')

plt.subplot(2,2,4)
plotfftreal(ret['sol'], fs)
plt.title('Recovered signal')

plt.show()
