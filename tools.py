import numpy as np
import matplotlib.pyplot as plt


def plotfftreal(s, fs):
    r"""
    Plot the Fourier transform in a nice way.

    Parameters
    ----------
    s : array_like
        The Fourier transform of a signal.
    fs : float
        The sampling frequency
    """
    N = len(s)
    w = np.linspace(0, fs-fs/N, N)

    plt.plot(w, np.abs(s), 'b.-')

    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')

#    plt.show()
