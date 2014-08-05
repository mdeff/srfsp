import numpy as np
import matplotlib.pyplot as plt


def plotfftreal(s, fs, stem=False):
    r"""
    Plot the Fourier transform in a nice way.

    Parameters
    ----------
    s : array_like
        The Fourier transform of a signal.
    fs : float
        The sampling frequency
    stem : bool, optional
        Use stem plot if ``True``, normal plot otherwise.
    """
    N = len(s)
    w = np.linspace(0, fs-fs/N, N)

    if stem:
        plot = plt.stem
    else:
        plot = plt.plot

    plot(w, np.abs(s), 'b.-')

    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')

#    plt.show()
