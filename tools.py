import numpy as np
import matplotlib.pyplot as plt


def plotfftreal(s, fs, dataset, amp='abs'):
    r"""
    Plot the Fourier transform in a nice way.

    Parameters
    ----------
    s : array_like
        Fourier transform of a signal.
    fs : float
        Sampling frequency.
    dataset : string
        Dataset name.
    amp : {'abs', 'real', 'imag'}
        Type of amplitude to plot.
    """
    N = len(s)
    w = np.linspace(0, fs-fs/N, N)

    exec('y = np.%s(s)' % (amp,))

    plt.plot(w, y, 'b.-')

    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude (%s)' % (amp,))

    if dataset is 'myoglobin':
        plt.xlim(938.5e3, 941.5e3)

#    plt.show()
