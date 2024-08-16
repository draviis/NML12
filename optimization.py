# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 12:34:26 2024

@author: ravi
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 15:12:23 2024

@author: ravi
"""

import numpy as np
from scipy.linalg import cholesky, LinAlgError
#from joblib import Parallel, delayed
from joblib import Parallel, delayed
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import cmath

####################################################
from scipy.linalg import cholesky, LinAlgError

def solveNewton(g, H):
    nVars = H.shape[0]

    try:
        R = cholesky(H, lower=True)
        d = -np.linalg.solve(R.T, np.linalg.solve(R, g))
    except LinAlgError:
        H = H + np.eye(nVars) * max(0, 1e-12 - min(np.real(np.linalg.eigvals(H))))
        d = -np.linalg.solve(H, g)

    return d

################################
def initialStepLength(i, adjustStep, f, g, gtd, t, f_prev):
    if i == 1 or adjustStep == 0:
        t = 1
    else:
        t = min(1, 2 * (f - f_prev) / gtd)

    f_prev = f
    return t, f_prev

##############################

def isLegal(v):
    legal = np.sum(np.any(np.imag(v))) == 0 and np.sum(np.isnan(v)) == 0 and np.sum(np.isinf(v)) == 0
    return legal

##############################
def polyinterp(points, xminBound, xmaxBound):
    nPoints = points.shape[0]
    order = np.sum(np.sum((np.imag(points[:, 1:3]) == 0))) - 1
    #print(nPoints)
    #print(order)
    xmin = np.min(points[:, 0])
    xmax = np.max(points[:, 0])

    # Compute Bounds of Interpolation Area
    if xminBound is None:
        xminBound = xmin
    if xmaxBound is None:
        xmaxBound = xmax

    # Code for the most common case:
    #   - cubic interpolation of 2 points
    #     with function and derivative values for both

    if nPoints >= 2 and order == 3:
        minPos = np.argmin(points[:, 0])
        notMinPos = (minPos + 1) % nPoints
        #print(minPos)
        #print(notMinPos)
        #notMinPos = -minPos+1
        d1 = points[minPos, 2] + points[notMinPos, 2] - 3 * (points[minPos, 1] - points[notMinPos, 1]) / (
                points[minPos, 0] - points[notMinPos, 0])
        #print(d1)
        #print(minPos)
        #print(points[minPos,2])
        #d2 = np.sqrt(d1 ** 2 - points[minPos, 2] * points[notMinPos, 2])
        d2 = cmath.sqrt(d1 ** 2 - points[minPos, 2] * points[notMinPos, 2])
        #print(d2)
        if np.isreal(d2):
            t = points[notMinPos, 0] - (
                    points[notMinPos, 0] - points[minPos, 0]) * ((points[notMinPos, 2] + d2 - d1) / (
                    points[notMinPos, 2] - points[minPos, 2] + 2 * d2))
            minPos = min(max(t, xminBound), xmaxBound)
        else:
            minPos = (xmaxBound + xminBound) / 2
        return minPos  # Return just the number

    # Constraints Based on available Function Values
    A = np.zeros((0, order + 1))
    b = np.zeros((0, 1))
    for i in range(nPoints):
        if np.imag(points[i, 1]) == 0:
            constraint = np.zeros(order + 1)
            for j in range(order, -1, -1):
                constraint[order - j] = points[i, 0] ** j
            A = np.vstack([A, constraint])
            b = np.vstack([b, points[i, 1]])

    # Constraints based on available Derivatives
    for i in range(nPoints):
        if np.isreal(points[i, 2]):
            constraint = np.zeros(order + 1)
            for j in range(1, order + 1):
                constraint[j - 1] = (order - j + 1) * points[i, 0] ** (order - j)
            A = np.vstack([A, constraint])
            b = np.vstack([b, points[i, 2]])

    # Find interpolating polynomial
    params = np.linalg.lstsq(A, b, rcond=None)[0]

    # Compute Critical Points
    dParams = np.zeros(order)
    for i in range(len(params) - 1):
        dParams[i] = params[i] * (order - i + 1)

    cp = np.hstack([xminBound, xmaxBound, points[:, 0], np.roots(dParams)]).T

    # Test Critical Points
    fmin = np.inf
    minPos = (xminBound + xmaxBound) / 2  # Default to Bisection if no critical points valid
    for xCP in cp:
        if np.isreal(xCP) and xminBound <= xCP <= xmaxBound:
            fCP = np.polyval(params, xCP)
            if np.isreal(fCP) and fCP < fmin:
                minPos = np.real(xCP)
                fmin = np.real(fCP)

    return minPos  # Return just the number
###################################################
def ArmijoBacktrack(x, t, d, f, fr, g, gtd, c1, LS_interp, LS_multi, progTol, debug, saveHessianComp, funObj, *varargin):

    # Evaluate the Objective and Gradient at the Initial Step
    
    f_new, g_new, H = funObj(x + t*d, *varargin)
    funEvals = 1

    while f_new > fr + c1 * t * gtd or not isLegal(f_new):
        temp = t

        if LS_interp == 0 or not isLegal(f_new):
            # Ignore value of new point
            if debug:
                print('Fixed BT')
            t = 0.5 * t
        elif LS_interp == 1 or not isLegal(g_new):
            # Use function value at new point, but not its derivative
            if funEvals < 2 or LS_multi == 0 or not isLegal(f_prev):
                # Backtracking w/ quadratic interpolation based on two points
                if debug:
                    print('Quad BT')
                #print(np.array([[0, f, gtd], [t, f_new, np.sqrt(-1)]]))    
                t = polyinterp(np.array([[0.0, complex(f), complex(gtd)], [complex(t), complex(f_new), cmath.sqrt(-1)]], dtype=np.complex128), 0, complex(t))
            else:
                # Backtracking w/ cubic interpolation based on three points
                if debug:
                    print('Cubic BT')
                t = polyinterp(np.array([[0.0, complex(f), complex(gtd)], [complex(t), complex(f_new), cmath.sqrt(-1)], [complex(t_prev), complex(f_prev), cmath.sqrt(-1)]]), 0, complex(t))
        else:
            # Use function value and derivative at new point
            if funEvals < 2 or LS_multi == 0 or not isLegal(f_prev):
                # Backtracking w/ cubic interpolation w/ derivative
                if debug:
                    print('Grad-Cubic BT')
                #print(np.array([[0.0, f, gtd], [t, f_new, np.dot(g_new.T, d)[0, 0]]], dtype=np.complex128))
                #print(np.array([[0.0, complex(f), complex(gtd)], [complex(t), complex(f_new), complex(np.dot(g_new.T, d)[0, 0])]], dtype=np.complex128))
                t = polyinterp(np.array([[0.0, complex(f), complex(gtd)], [complex(t), complex(f_new), complex(np.dot(g_new.T, d)[0, 0])]], dtype=np.complex128), 0, complex(t))
            elif not isLegal(g_prev):
                # Backtracking w/ quartic interpolation 3 points and derivative of two
                if debug:
                    print('Grad-Quartic BT')
                t = polyinterp(np.array([[0.0, complex(f), complex(gtd)], [complex(t), complex(f_new), complex(np.dot(g_new.T, d)[0, 0])], [complex(t_prev), complex(f_prev), cmath.sqrt(-1)]]), 0, complex(t))
            else:
                # Backtracking w/ quintic interpolation of 3 points and derivative of two
                if debug:
                    print('Grad-Quintic BT')
                t = polyinterp(np.array([[0.0, complex(f), complex(gtd)], [complex(t), complex(f_new), complex(np.dot(g_new.T, d)[0, 0])], [complex(t_prev), complex(f_prev), complex(np.dot(g_prev.T, d)[0,0])]]), 0, complex(t))

        # Adjust if change in t is too small/large
        if t < temp * 1e-3:
            if debug:
                print('Interpolated Value Too Small, Adjusting')
            t = temp * 1e-3
        elif t > temp * 0.6:
            if debug:
                print('Interpolated Value Too Large, Adjusting')
            t = temp * 0.6

        # Store old point if doing three-point interpolation
        if LS_multi:
            f_prev = f_new
            t_prev = temp
            if LS_interp == 2:
                g_prev = g_new

        
        f_new, g_new, H = funObj(x + t * d, *varargin)
        
        funEvals += 1

        # Check whether step size has become too small
        if np.max(np.abs(t * d)) <= progTol:
            if debug:
                print('Backtracking Line Search Failed')
            t = 0
            f_new = f
            g_new = g
            break

    # Evaluate Hessian at new point
    if funEvals > 1 and saveHessianComp:
        f_new, g_new, H = funObj(x + t * d, *varargin)
        funEvals += 1

    x_new = x + t * d


    return t, x_new, f_new, g_new, funEvals, H

###########################################
def noProgress(td, f, f_old, optTol, verbose):
    x = 0

    if abs(f - f_old) < optTol:
        x = 1
        if verbose:
            print('Change in Objective below optTol')

    elif np.sum(np.abs(td)) < optTol:
        x = 1
        if verbose:
            print('Step Size below optTol')

    return x

#######################################
def process_options(*args):
    options = {}  # Initialize an empty dictionary to store options

    # Loop through the input arguments
    for i in range(0, len(args), 2):
        # Extract the option name and value
        option_name = args[i]
        option_value = args[i + 1]

        # Store the option in the dictionary
        options[option_name] = option_value

    return options

##########################################
# def l1GeneralSmooth_sub0(gradFunc, w, G, *args):
#     options_dict = process_options('verbose', 0, 'threshold', 1e-5,
#                                    'optTol', 1e-5, 'maxIter', 500, 'alpha', 5e4,
#                                    'update1', 1.25, 'update2', 1.5, 'adjustStep', 1, 'predict', 0)
#     verbose, threshold, optTol, maxIter, alpha, update1, update2, adjustStep, predict = \
#         [options_dict[key] for key in ['verbose', 'threshold', 'optTol', 'maxIter', 'alpha', 'update1', 'update2', 'adjustStep', 'predict']]

#     if verbose:
#         print('{:10s} {:10s} {:15s} {:15s} {:15s} {:8s} {:15s}'.format(
#             'Iteration', 'FunEvals', 'Step Length', 'Function Val', 'Opt Cond', 'Non-Zero', 'Alpha'))

#     p = len(w)
#     alpha_init = 1
#     currParam = alpha_init

#     f, g, H = gradFunc(w, currParam, G, *args)
#     fEvals = 1
#     t = 1
#     f_prev = f

#     for i in range(1, maxIter + 1):
#         w_old = w
#         f_old = f
#         d = solveNewton(g, H)
#         gtd = np.dot(g.T, d)

#         if gtd > -optTol:
#             if verbose:
#                 print('Directional Derivative too small')
#             break

#         t, f_prev = initialStepLength(i, adjustStep, f, g, gtd, t, f_prev)
#         t, w, f, g, LSfunEvals, H = ArmijoBacktrack(
#             w, t, d, f, f, g, gtd, 1e-4, 2, 0, optTol, max(verbose - 1, 0), 0, gradFunc, currParam, G, *args)
#         fEvals += LSfunEvals

#         if verbose:
#             print('{:10d} {:10d} {:15.5e} {:15.5e} {:15.5e} {:8d} {:15.5e}'.format(
#                 i, fEvals, t, f, np.sum(np.abs(g[np.abs(w) >= threshold])),
#                 np.sum(np.abs(w) > threshold), currParam))

#         oldParam = currParam
#         if LSfunEvals == 1:
#             currParam = min(currParam * update2, alpha)
#         else:
#             currParam = min(currParam * update1, alpha)

#         if verbose == 2 and currParam >= alpha:
#             print('At max alpha')

#         if np.sum(np.abs(g[np.abs(w) >= threshold])) < optTol and oldParam == alpha:
#             if verbose:
#                 print('Solution Found')
#             break

#         if noProgress(t * d, f, f_old, optTol, verbose):
#             break
#         elif fEvals > maxIter:
#             break

#         if predict and currParam != oldParam and i % 3 == 0:
#             sig = (1 + np.exp(oldParam * w))**-1
#             lambda_val = args[1]
#             g_alpha = 2 * lambda_val * w * sig * (1 - sig)

#             predictDir = -np.linalg.solve(H, g_alpha)
#             w = w + (currParam - oldParam) * predictDir
#             f, g, H = gradFunc(w, currParam, G, *args)
#             fEvals += 1

#     w[np.abs(w) < threshold] = 0

#     return np.real(w)

##############################
###############################


def predict(coefficients, X, model_type='linear', threshold=0.5):
    linear_predictor = np.dot(X, coefficients)

    if model_type == 'linear':
        return linear_predictor
    elif model_type == 'logistic':
        probabilities = 1 / (1 + np.exp(-linear_predictor))
        binary_predictions = (probabilities >= threshold).astype(int)
        return binary_predictions
    elif model_type == 'poisson':
        predicted_counts = np.exp(linear_predictor)
        return predicted_counts
    else:
        raise ValueError("Invalid model type. Supported types: 'linear', 'logistic', 'poisson'")

def deviance_loss(y, mu, family="gaussian"):
    if family == "gaussian":
        return (y - mu) ** 2
    elif family == "binomial":
        return y * mu - np.log(1 + np.exp(mu))
    elif family == "poisson":
        return y * np.log(mu) - mu

def loss_func(test_true, test_pred,type_measure='mse'):
    if type_measure == "mse":
        return np.average((test_true - test_pred)**2)#, weights=w)
    elif type_measure == "mae":
        return np.average(np.abs(test_true - test_pred))#, weights=w)
    elif type_measure == "class":
        return np.average(np.round(test_pred) == test_true)#, weights=w)
    elif type_measure == "deviance":
        return np.average(deviance_loss(test_true, test_pred, family='gaussian'))#, weights=w)
    elif type_measure == "poisson":
        np.mean(test_pred - test_true * np.log(test_pred + 1e-9))
    else:
        raise ValueError(f"Loss type {type_measure} not implemented.")
        
        
        
        
def logspace(x, y, length_out):
    return np.exp(np.linspace(np.log(x), np.log(y), num=length_out))

def lasso_preprocessing(X, y,offset=None, weights=None, nlambda=10, intercept=1):
    nobs, nvars = X.shape
    lambda_min_ratio = 0.1 if nobs < nvars else 1e-04
    
    if offset is None:
        offset = np.zeros(nobs)
    
    nlambda = int(nlambda)
    
    Xsc = X.copy()
    X_scale = np.ones(nvars)
    X_center = np.zeros(nvars)
    
    if weights is None:
        weights = np.ones(nobs)
    
    lambda_values = None  # Initialize lambda_values
    
    if lambda_values is None:
        # Assuming y is defined somewhere, if not replace it with your actual target variable
        #y = np.random.rand(nobs)
        
        lambda_max = np.max(np.abs(np.dot(Xsc.T, y - offset - np.average(y, weights=weights, axis=0) * intercept) / nobs))
        lambda_values = logspace(lambda_min_ratio * lambda_max, lambda_max, length_out=nlambda)
    
    return lambda_values