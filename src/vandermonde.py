
#!/usr/bin/env python
# coding: utf-8

import numpy as np
from numpy.linalg import lstsq

"""
    The general form for n-dimensions (x1, x2, ..., xn) is as follows:
        V = | 1 x1 x1^2 . . . x1^(n-1) x1^n |
            | 1 x2 x2^2 . . . x2^(n-1) x2^n |
            | 1 x3 x3^2 . . . x3^(n-1) x3^n |
            | :  :   :           :       :  |
            | :  :   :           :       :  |
            | 1 xn xn^2 . . . xn^(n-1) xn^n |
    Note: You'll see the general form of the equation change depending on the reference, hence why I am outlining
    the whole thing here. I am following a combination of them including:
    
            http://pages.cs.wisc.edu/~sifakis/courses/cs412-s13/lecture_notes/CS412_12_Feb_2013.pdf
            https://en.wikipedia.org/wiki/Vandermonde_matrix
            
    Our Vandermonde matrix involves 3 dimensions which we will call (x, y, z).
        V = | x x^2 . . . x^(n-1) x^n   ------------0------------ ------------0------------ |
            | ------------0------------  y y^2 . . . y^(n-1) y^n  ------------0------------ |
            | ------------0------------ ------------0------------  z z^2 . . . z^(n-1) z^n  |
    This is a linear set of equations to solve to get a new set of coordinates (x', y', z'):
            | x x^2 . . . x^(n-1) x^n   ------------0------------ ------------0------------ | | c0 |   | x' |
            | ------------0------------  y y^2 . . . y^(n-1) y^n  ------------0------------ | | :  | = | y' |
            | ------------0------------ ------------0------------  z z^2 . . . z^(n-1) z^n  | | cn |   | z' |
            
    --> We have a general  idea of what our starting coeficients should be, which reduces the computational cost.
    --> We are solving for 27 coefficients, (c1, c2, c3, ..., c27). Our startung point will be a form of the identity matrix,
        since we know that our correction to the data point should not shift it that drastically. If it does, we throw out
        the correction
    --> Numpy does have a dedicated Vandermonde matrix, but it differs slightly from what we are trying to do here which is why
        I'm not using it.
"""
class Vandermonde():

    def __init__():
        # We don't use this yet, but I am throwing it in here for later
        self.polyorder = 27
        # Initialize the coefficient matrix
        self.coefficients = np.zeros(27)
        self.coefficients[0]  = 1.0
        self.coefficients[10] = 1.0
        self.coefficients[20] = 1.0
        # Set the Vandermonde matrix to a zero matrix
        self.vandermonde = np.zeros((3,27))
    

    def update_coefficients( self, r, r_, intensity):
        """
            To update the coefficients based on new points
            @paras:
            r : nx3 numpy array of actual locations of the points of the form: r  = (x,y,z)
            r_: nx3 numpy array of ideal locations of the points of the form: r_  = (x_,y_,z_)
            intensity : nx1 numpy array of intensities of each point
            @returns:
            T: final homogeneous transformation that maps A on to B
            distances: Euclidean distances (errors)
        """
        assert r.shape == r_.shape
        
        # Flatten the arrays
        weights = np.power( intensity, 2).flatten
        B = r_.flatten
        [ A.append( self.get_matrix( i ) ) for i in r ]
        self.linear_least_squares(A,B,self.coefficients,weights)
        # This would be a good spot to write out the new coefficients
        for j in self.coefficients:
            print ( j )

    def get_matrix( self, r ):
        """
            To get the Vandermonde Matrix
            @paras:
            r : 1x3 numpy array of actual locations of the points of the form: r  = (x,y,z)
            x : 1st value of r
            y : 2nd value of r
            z : 3rd value of r
            dv : the Vandermonde matrix component along the diagonal
            @returns:
            vandermonde: 3xpolyorder Vandermonde matrix
        """
        x = r[0]
        y = r[1]
        z = r[2]
        # Lambda functions are my favorite and I have an excuse to use them here.
        power = lambda n: n * n
        cross = lambda n1, n2: n1 * n2
        # Probably not the most elegant way to do this, but I'll clean it up later
        dv = [x, y, z, power(x), power(y), power(z), cross(x,y), cross(x,z), cross(y,z)]
        self.vandermonde[0,0:8]   = dv
        self.vandermonde[1,9:17]  = dv
        self.vandermonde[2,18:26] = dv
        # return the Vandermonde matrix
        return self.vandermonde

    def linear_least_squares( self, A, B, weights ):
        """
            To get the linear least squre fit, solve Ax = B
            @paras:
            A : A matrix
            B : B matrix
            weights : the weight of each data point
            wA : the weighted A matrix
            wB : the weighter B matrix
            @returns:
            coefficients: solved for x
            residuals
        """
        wA = np.multiply(weights, A)
        wB = np.multiply(weights, B)
        assert wA.shape[0] == wB.shape[0]
        polycoefficients, residuals = lstsq(wA, wB, rcond=None)
        # If the new coefficients are not significantly larger than the previous,
        # then accept the change
        if (  abs( polycoefficients - self.coefficients) < 1 ):
            self.coefficients = polycoefficients
        else:
            raise Exception ( 'Error: Data is not fitting' )
        
        return self.coefficients, residuals
        
    def correct_data( self ):
        """
            To get the corrected data, solve Ax = B
            @paras:
            Vandermonde : matrix, equivalent to A
            coefficient : equivalent to x
            @returns:
            corrected_data: equivalent to B
        """
        corrected_data = np.matmul( self.vandermonde, self.coefficients )
        return corrected_data




