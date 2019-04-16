#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def mean_distance(A, B, axis):
    """
    To calculate the mean Euclidean distance between two set of points along 
    specified axis.
    @paras:
        A, B: two set of points.
        aixs: specified axis.
    @return:
        the mean Euclidean distance.
    """
    assert A.shape == B.shape
    return np.mean(np.linalg.norm(A-B, axis=axis))
    

def best_fit_transform(A, B):
    """
    To calculates the least-squares best-fit transform that maps corresponding 
    points A to B in m spatial dimensions
    @paras:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    @returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    """

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t

def icp(A, B, init_pose=None, max_iterations=20, tolerance=0.001):
    """
    The Iterative Closest Point method: finds best-fit transform that maps 
    points A on to points B
    @paras:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    @returns:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors)
        i: number of iterations to converge
    """

    assert A.shape == B.shape

    # get number of dimensions
    N, m = A.shape[0], A.shape[1]
    

    # make points homogeneous, copy them to maintain the originals
    src = np.vstack((A.T, np.ones(N)))
    dst = np.vstack((B.T, np.ones(N)))

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        # compute the transformation between the current source and nearest destination points
        T,_,_ = best_fit_transform(src[:m].T, dst[:m].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        distance = mean_distance(src[:m], dst[:m], axis=0)
        if np.abs(prev_error - distance) < tolerance:
            break
        prev_error = distance

    # calculate final transformation
    T,_,_ = best_fit_transform(A, src[:m].T)

    return T, distance, i