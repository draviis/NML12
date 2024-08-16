# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 13:19:54 2024

@author: ravi
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from collections import defaultdict
import matplotlib.pyplot as plt
np.random.seed(1)

class eLasso:
    def __init__(self, alpha = np.logspace(-2,4,10).tolist(), group=None,max_iter: int = 1000, fit_intercept: bool = True,tolerance=10e-5) -> None:
        if type(alpha)==list:
            self.alpha=alpha
        else:
            self.alpha= [alpha]        
        self.group = group
        self.max_iter: int = max_iter 
        self.fit_intercept: bool = fit_intercept 
        self.tolerance = tolerance
        self.coef_ = None 
        self.intercept_ = None         
        
    
        b0 = list(range(1, len(self.group) + 1))
        unique_values = list(set(self.group))
        self.groups = [[0 for _ in range(len(self.group))] for _ in range(len(unique_values))]
        for i, value in enumerate(b0):
            self.group_index = unique_values.index(self.group[i])
            self.groups[self.group_index][i] = value        
        m = []
        z = list(np.zeros([len(b0)]))        
        for i in range(len(self.groups)):
            num = len([element for element in self.groups[i] if element != 0]) - 1
            zerorows = [z] * num
            m.append(np.array([self.groups[i]] + zerorows))    
        G = np.vstack(m) 
        self.G = np.vstack([[1 if element != 0 else 0 for element in sublist] for sublist in G])   
        
    def _split_intercept(self,beta):
        return beta[0], beta[1:]         
    
        
    def _gradhess(self,X,y,beta):
        gradient2 = [[] for j in range(len(self.alpha))]
        hessian2 = [[] for j in range(len(self.alpha))]
        for j in range(len(self.alpha)): 
            if self.fit_intercept:
                _,b = self._split_intercept(beta[j])
            else:
                b = beta[j]
            #print(len(b))
            c= 10e-5
            A1 = self.G.T@self.G            
            A2 = A1+A1.T
            B = [[] for i in range(len(b))]    
            grad = [[] for i in range(len(b))]
            hess = [[] for i in range(len(b))]    
            for i in range(len(b)):
                B[i] = np.sqrt(b[i]**2+c)      
            for i in range(len(b)):
                grad[i] = b[i]/(np.sqrt(b[i]**2+c))
                hess[i] = c/(np.sqrt(b[i]**2+c)**3)    
            grad1 = B@(A1+A1.T)*grad
            hess1 = (grad1@A2.T@grad1+B@A2@hess)*np.eye(len(b))
            if self.fit_intercept:
                grad1 = np.hstack((0, grad1))
                hess1 = np.hstack((np.zeros((hess1.shape[0], 1)),hess1))
                hess1 = np.vstack((np.zeros((1, hess1.shape[1])), hess1))
            gradient2temp = -X.T@(y-X@beta[j])+self.alpha[j]*grad1
            gradient2[j]=gradient2temp
            hessian2temp = X.T@X +self.alpha[j]*hess1
            hessian2[j]=hessian2temp
            #print(gradient2temp)
            #print(hessian2temp)
        return gradient2,hessian2           
    
        
    def fit(self,X,y):
        if self.fit_intercept:
            X = np.column_stack((np.ones(len(X)), X))        
        #betas = []                   
        beta = [np.zeros(X.shape[1]) for w in range(len(self.alpha))]              
        #print(np.array(beta).shape)
        for iteration in range(self.max_iter):
        # Get gradient and hessian
            gradient,hessian = self._gradhess(X,y,beta)         
                              
            # Newton method iteration scheme
            for i in range(len(self.alpha)): 
                #print(hessian[i])
                #print(len(gradient[i]))
                beta[i] -= np.linalg.inv(hessian[i]) @ gradient[i]
                #lr *= lr_decr
                #print(x_star[i+1])
                #print(x_star[i])
                # Check convergence criteria
                if np.linalg.norm(gradient[i])  < self.tolerance:
                   break 

 
        if self.fit_intercept:
            self.intercept_ = [inner_list[0] for inner_list in beta]
            self.coef_ =  [inner_list[1:] for inner_list in beta]
        else:
            self.coef_ = beta
        
        return self
        
    
    def predict(self, X: np.ndarray):
        y = [np.dot(X, self.coef_[i]) for i in range(len(self.alpha))]
        if self.fit_intercept:
            for i in range(len(self.alpha)): 
                y[i] += self.intercept_[i] * np.ones(len(y[i]))
        return y
    
    def _mse(self,y_true, y_pred):
        return mean_squared_error(y_true, y_pred)
    
    def cv(self,X, y, k=5):
        kf = KFold(n_splits=k, shuffle=True, random_state=24)
        #self.mse_all = [[] for j in range(len(self.alpha))]
        self.mse_all = []
        #print(len(self.mse_all))
        for train_index, val_index in kf.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            #print(X_train.shape)
            #print(X_val.shape)
            #print(y_train.shape)
            #print(y_val.shape)

            self.fit(X_train,y_train)
            #print(len(self.coef_))
            #print(self.coef_)
            #print(self.intercept_)
            if self.fit_intercept:
                X_val = np.column_stack((np.ones(len(X_val)), X_val)) 
                betas = [np.concatenate(([intercept], coef)) for intercept, coef in zip(self.intercept_, self.coef_)]
                y_pred = [np.dot(X_val, betas[i]) for i in range(len(betas))]
            else:
                y_pred = [np.dot(X_val, self.coef_[i]) for i in range(len(self.coef_))]
            
            #y_pred = self.predict(X_val)
            #print(len(y_pred))
            #print(y_pred)
            msetemp = [self._mse(y_val,y_pred[i]) for i in range(len(y_pred))]
            #print(msetemp)    
            self.mse_all.append(msetemp)
        #print(self.mse_all)
        #print(np.mean(self.mse_all,axis=0))
        self.arr = np.mean(self.mse_all,axis=0)
        min_index = np.argmin(self.arr)
        #print("Index of the minimum value:", min_index)
        #print("Minimum value:", self.arr[min_index])
        print("Best alpha value:",self.alpha[min_index])
        
    
    def plot(self): 
        results_array= np.array(self.coef_)
        indices = defaultdict(list)
        for i, v in enumerate(self.group):
          indices[v].append(i)
        group_list = [value for key, value in indices.items()]
        # Choose a colormap (e.g., 'viridis', 'plasma', 'cividis', 'magma', 'inferno', 'twilight', etc.)
        colormap = plt.get_cmap('magma')
        # Generate 3 equally spaced colors from the colormap
        colors = [colormap(i) for i in np.linspace(0.1, 0.8, len(set(self.group)))]
        plt.figure(figsize=(10, 6))
        # Loop over each group and plot the paths for all coefficients in the group
        for i, group1 in enumerate(group_list):
            # Pick a color for the group
            group_color = colors[i]
            for j in group1:
                plt.plot(self.alpha, results_array[:, j], label=f'Coefficient {j + 1} - Group {i + 1}', color=group_color)

        plt.xscale('log')  # Set x-axis to logarithmic scale for better visualization
        plt.xlabel('Lambda (Regularization Strength)')
        plt.ylabel('Coefficient Value')
        plt.title('Regularization Path Plot')
        #plt.legend()
        plt.grid(True)
        plt.show()

########################################################################
###########################################################################        
N = 100  # number of observations
p = 5  # number of variables

# Randomly generated X
X = np.random.normal(size=(N, p))

#Standardization: mean = 0, std = 1
X = (X - X.mean(axis=0)) / X.std(axis=0)
#Y  = (Y - Y.mean(axis=0)) / Y.std(axis=0)
# Artificial coefficients
betao = np.array([1,2,-4,-5,3])

# Y variable, standardized Y
#y = 3+X @ betao + np.random.normal(scale=0.5, size=N)
y = .67+betao[0]*(X[:,0])+betao[1]*X[:,1]+betao[2]*X[:,2]+betao[3]*(X[:,3])+betao[4]*(X[:,4])

group = [1, 1, 2,2,2]   

lambda_values1 = np.logspace(-2,4,10)            
model = eLasso(group=group,alpha=.10000,
                fit_intercept=True,tolerance=10e-5)  

model.fit(X,y)
#model.cv(X,y,k=5) 
model.plot()
model = eLasso(group=group,alpha=21.5,
                fit_intercept=False,tolerance=10e-5) 
model.fit(X,y)
model.coef_  

#####################################################
####################################################

import numpy as np
def logspace(x, y, length_out):
    return np.exp(np.linspace(np.log(x), np.log(y), num=length_out))

def lasso_preprocessing(X, offset=None, weights=None,  nlambda=100, intercept=1):
    nobs, nvars = X.shape
    lambda_min_ratio = 0.01 if nobs < nvars else 1e-04

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
        lambda_max = np.max(np.abs(np.dot(Xsc.T, y - offset - np.average(y, weights=weights) * intercept) / nobs))
        lambda_values = logspace(lambda_min_ratio * lambda_max, lambda_max, length_out=nlambda)

    return lambda_values

# Example usage:
# Assuming you have X and other necessary variables defined
lambda_values = lasso_preprocessing(X)

