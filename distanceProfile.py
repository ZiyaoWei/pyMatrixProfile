import numpy as np
from util import *

def naiveDistanceProfile(tsA, idx, m, tsB = None):
    """Return the distance profile of query against ts. Use the naive all pairs comparison algorithm.

    >>> np.round(naiveDistanceProfile(np.array([0.0, 1.0, -1.0, 0.0]), 0, 4, np.array([-1, 1, 0, 0, -1, 1])), 3)
    array([[ 2.   ,  2.828,  2.   ],
           [ 0.   ,  0.   ,  0.   ]])
    """
    selfJoin = False
    if tsB is None:
        selfJoin = True
        tsB = tsA

    query = tsA[idx : (idx + m)]
    distanceProfile = []
    n = len(tsB)
    for i in range(n - m + 1):
        distanceProfile.append(zNormalizedEuclideanDistance(query, tsB[i : i + m]))
    if selfJoin:
        trivialMatchRange = (max(0, idxToProcess - m / 2), min(idxToProcess + m / 2 + 1, len(tsB)))
        distanceProfile[trivialMatchRange[0] : trivialMatchRange[1]] = np.inf
    return (distanceProfile, np.full(n - m + 1, idx, dtype = float))


def stampDistanceProfile(tsA, idx, m, tsB = None):
    """
    >>> np.round(stampDistanceProfile(np.array([0.0, 1.0, -1.0, 0.0]), 0, 4, np.array([-1, 1, 0, 0, -1, 1])), 3)
    array([[ 2.   ,  2.828,  2.   ],
           [ 0.   ,  0.   ,  0.   ]])
    """
    selfJoin = False
    if tsB is None:
        selfJoin = True
        tsB = tsA

    query = tsA[idx : (idx + m)]
    n = len(tsB)
    distanceProfile = mass(query, tsB)
    if selfJoin:
        trivialMatchRange = (max(0, idxToProcess - m / 2), min(idxToProcess + m / 2 + 1, len(tsB)))
        distanceProfile[trivialMatchRange[0] : trivialMatchRange[1]] = np.inf
    return (distanceProfile, np.full(n - m + 1, idx, dtype = float))

if __name__ == "__main__":
    import doctest
    doctest.testmod()
