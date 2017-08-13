# pyMatrixProfile
Python implementation of matrix profile algorithms (http://www.cs.ucr.edu/~eamonn/MatrixProfile.html).

Currently only STAMP is implemented. STOMP and SCRIMPT are coming soon, stay tuned!

There should be an API redesign coming - the usability isn't very good now, but you can run


    >>> np.round(stmp(tsA = np.array([0.0, 1.0, -1.0, 0.0, 0.0]), tsB = np.array([-1, 1, 0, 0, -1, 1]), m = 4), 3)
    array([[ 2.,  2.,  2.],
           [ 0.,  1.,  0.]])

If you use this work, please consider citing the following paper, which this repository is based on:

* Chin-Chia Michael Yeh, Yan Zhu, Liudmila Ulanova, Nurjahan Begum, Yifei Ding, Hoang Anh Dau, Diego Furtado Silva, Abdullah Mueen, Eamonn Keogh (2016). Matrix Profile I: All Pairs Similarity Joins for Time Series: A Unifying View that Includes Motifs, Discords and Shapelets. IEEE ICDM 2016.
