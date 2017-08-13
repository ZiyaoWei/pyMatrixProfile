import numpy as np
import numpy.fft as fft


def zNormalize(ts):
    """Return a z-normalized version of the time series ts.

    >>> zNormalize(np.array([1.0, 1.0, 1.0]))
    array([ 0.,  0.,  0.])
    >>> np.round(zNormalize(np.array([1.0, 2.0, 0.0])), 3)
    array([ 0.   ,  1.225, -1.225])
    >>> np.round(zNormalize(np.array([0.2, 2.2, -1.8])), 3)
    array([-0.   ,  1.225, -1.225])
    """
    ts -= np.mean(ts)
    stdev = np.std(ts)
    if stdev <> 0:
        ts /= stdev
    return ts

def zNormalizedEuclideanDistance(tsA, tsB):
    """Return the z-normalized Euclidean Distance between tsA and tsB.

    >>> zNormalizedEuclideanDistance(np.array([1.0, 1.0, 1.0]), np.array([2.0, 2.0, 2.0]))
    0.0
    >>> zNormalizedEuclideanDistance(np.array([1.0, 1.0, 1.0]), np.array([2.0, 2.0]))
    Traceback (most recent call last):
        ...
    ValueError: tsA and tsB must be of the same length
    >>> np.round(zNormalizedEuclideanDistance(np.array([0.0, 2.0, -2.0, 0.0]), np.array([1.0, 5.0, 3.0, 3.0])), 3)
    2.0
    """
    if len(tsA) <> len(tsB):
        raise ValueError("tsA and tsB must be of the same length")
    return np.linalg.norm(zNormalize(tsA.astype("float64")) - zNormalize(tsB.astype("float64")))

def movstd(ts, m):
    """
    >>> np.round(movstd(np.array([1, 2, 3, 10]), 3), 3)
    array([ 0.816,  3.559])
    """
    if m < 1:
        raise ValueError("Query length m must be >= 1")

    ts = ts.astype("float")
    s = np.insert(np.cumsum(ts), 0, 0)
    sSq = np.insert(np.cumsum(ts ** 2), 0, 0)
    segSum = s[m:] - s[:-m]
    segSumSq = sSq[m:] - sSq[:-m]
    return np.sqrt(segSumSq / m - (segSum / m) ** 2)

def mass(query, ts):
    """
    >>> np.round(mass(np.array([0.0, 1.0, -1.0, 0.0]), np.array([-1, 1, 0, 0, -1, 1])), 3)
    array([ 2.   ,  2.828,  2.   ])
    """
    query = zNormalize(query)
    m = len(query)
    n = len(ts)

    stdv = movstd(ts, m)
    query = query[::-1]
    query = np.pad(query, (0, n - m), 'constant')
    dots = fft.irfft(fft.rfft(ts) * fft.rfft(query))
    return np.sqrt(2 * (m - (dots[m - 1 :] / stdv)))

if __name__ == "__main__":
    import doctest
    doctest.testmod()
