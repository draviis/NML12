
import numpy as np
from scipy.linalg import cholesky, LinAlgError
#from joblib import Parallel, delayed
from joblib import Parallel, delayed
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import cmath
from optimization import solveNewton, initialStepLength, ArmijoBacktrack, noProgress, process_options,predict,loss_func,lasso_preprocessing
import time
from scipy.special import logsumexp
import scipy
from scipy.linalg.blas import sgemm as matmm
from scipy.linalg.blas import sgemv as matmv
class nml121:
    def __init__(self, lmd, family, group=None,max_iter: int = 1000, fit_intercept: bool = True) -> None:
        t1=time.time()
        self.group = group
        self.max_iter: int = max_iter 
        self.fit_intercept: bool = fit_intercept 
        #self.tolerance = tolerance
        self.coef_ = None 
        self.intercept_ = None   
        self.lmd = lmd
        self.family  = family
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
        t2=time.time()
        self.init_time=t2-t1
        
    def _split_intercept(self,beta):
        return beta[0], beta[1:]   
    
    def _mylogsumexp(self,beta):
        B = np.max(beta, axis=1, keepdims=True)
        #lse = np.log(np.sum(np.exp(beta - B), axis=1, keepdims=True)) + B
        lse = logsumexp(beta-B,axis=1,keepdims=True) + B
        return lse
    def _linear_mse(self,beta, X, y):
        residuals = y - np.dot(X, beta)    
        mse = np.sum(residuals**2) 
        #matmm
        g = -matmm(alpha=1.0,a=X.T,b=(y-X@beta))
        H = matmm(alpha=1.0,a=X.T,b=X)
        #g = -np.dot(X.T,(y-X@beta))
        #H = np.dot(X.T,X) 
        return mse,g,H
    def _logistic_loss(self,beta, X, y):
        n, p = X.shape    
        Xw = np.dot(X, beta)
        yXw = y * Xw
        
        nll = np.sum(self._mylogsumexp(np.hstack([np.zeros((n, 1)), -yXw])))
        sig = 1 / (1 + np.exp(-yXw))
        g = -matmm(1.0,X.T, y * (1 - sig))
        #np.dot(X.T, np.dot(np.diagflat(sig * (1 - sig)), X))
        H = matmm(alpha=1.0,a=X.T, b=matmm(alpha=1.0,a=np.diagflat(sig * (1 - sig)), b=X))
        #probabilities = sigmoid(Xw)
        probabilities = 1 / (1 + np.exp(-Xw))
        nll = -np.sum(y * np.log(probabilities) + (1 - y) * np.log(1 - probabilities))
        return nll, g, H
    def _poisson_loss(self,beta, X, y):
        lambdas = np.exp(np.clip(np.dot(X, beta), -700, 700))
        # #nll = -np.sum(y * np.log(lambdas) - lambdas)/n
        # nll = -np.sum(y * np.log(lambdas + 1e-15) - lambdas) / n
        # #l = np.exp(np.dot(X,w))
        # l = np.exp(np.dot(X, w)) + 1e-15
        # L = l*np.eye(len(X))
        # g = -X.T@(y-l)
        # H = np.dot(X.T,np.dot(L,X))
        #n = len(y)
        Xw = matmm(1.0,X, beta)
        #lambdas = np.exp(Xw)        
        # Negative log-likelihood using gammaln
        nll = -np.sum(y * Xw - lambdas)#- gammaln(y + 1))
        # Gradient
        g = -matmm(1.0,X.T , (lambdas - y))        
        # Hessian
        H = matmm(1.0,X.T , (np.diag(lambdas.flatten()) @ X))    
        return nll, g, H
    
    def _sigmoidL1(self,beta, alpha,G, family, lmd, *args):
        if self.family == 'gaussian':
            gradFunc = self._linear_mse
        elif self.family == 'binomial':
            gradFunc = self._logistic_loss
        elif self.family == 'poisson':
            gradFunc = self._poisson_loss
        t3=time.time()
        A1 = matmm(1.0,self.G.T,self.G)
        A2 = A1+A1.T 
        t4=time.time()
        self.G_time=t4-t3
        nll, g, h = gradFunc(beta, *args)
        
        if self.fit_intercept:
            _,w = self._split_intercept(beta)
        else:
            w = beta       
        
        p = len(w)
        t5=time.time()
        lsep = self._mylogsumexp(np.hstack([np.zeros((p, 1)), alpha*w]))
        lsem = self._mylogsumexp(np.hstack([np.zeros((p, 1)), -alpha*w]))
        B = ((lsep+lsem)/alpha).flatten()
        t6=time.time()
        self.logsumexp_time=t6-t5
        
        t7=time.time()
        grad_matrix = (1 - 2 * np.exp(-lsep)).reshape(-1)
        #print(grad_matrix.shape)
        #hess_matrix = (np.exp(np.log(2) + np.log(alpha) + alpha*w - 2*lsep)).flatten().tolist()
        hess_matrix = (np.exp(np.log(2) + np.log(alpha) + alpha*w - 2*lsep)).flatten()
        #print(hess_matrix.shape)
        # print(B.shape)
        #print(A2.T.shape)
        grad_matrix1 = matmv(1.0,B,matmv(1.0,(A1+A1.T),grad_matrix))         
        hess_matrix1 = ((grad_matrix@A2.T@grad_matrix + B@A2@hess_matrix)*np.eye(len(w)))
        #print(B@A2@hess_matrix)
        #print('a',(B@A2@hess_matrix).shape)
        #print('b',(grad_matrix@A2.T@grad_matrix).shape)
        #hess_matrix1 = (matmv(1.0,matmv(1.0,A2.T,grad_matrix),grad_matrix) + matmm(1.0,matmv(1.0,A2,B),hess_matrix))*np.eye(len(w))
        #temp1=matmv(1.0,matmv(1.0,A2.T,grad_matrix),grad_matrix)
        #print('c',temp1.shape)
        #temp2=matmv(1.0,hess_matrix,matmv(1.0,A2,B))
        #print('d',temp2.shape)
        #hess_matrix1 = (temp1+temp2)*np.eye(len(w))
        #hess_matrix1=matmv(1.0,np.eye(len(w)),(temp1+temp2))
        #print(hess_matrix1.shape)
        # temp1=matmv(1.0,matmm(1.0,A2.T,grad_matrix),grad_matrix)
        # temp2=matmv(1.0,matmm(1.0,A2,hess_matrix),B)
        # hess_matrix1=matmm(1.0,(temp1+temp2),np.eye(len(w)))
        
        # temp1=matmm(1.0,A2,hess_matrix)
        # temp2=B@temp1
        # temp3=grad_matrix@A2.T
        #temp4=matmv(1.0,temp3,gr,ad_matrix)
        #hess_matrix1=(temp4+temp2)*np.eye(len(w))
        
        #hess_matrix1 = (matmm(1.0,grad_matrix,matmm(1.0,A2.T,grad_matrix)) + matmv(matmm(1.0,A2,hess_matrix),B))*np.eye(len(w))
        #matmm(1.0,grad_matrix,matmm(1.0,A2.T,grad_matrix))
        #matmv(matmm(1.0,A2,hess_matrix),B)
        
        
        if self.fit_intercept:
            grad_matrix1 = np.hstack((0, grad_matrix1))
            hess_matrix1 = np.hstack((np.zeros((hess_matrix1.shape[0], 1)),hess_matrix1))
            hess_matrix1 = np.vstack((np.zeros((1, hess_matrix1.shape[1])), hess_matrix1))           
        
        g = g + self.lmd*(grad_matrix1).reshape(len(beta),1)
        h= h + self.lmd*hess_matrix1
        nll = nll + self.lmd*(B@G.T@G@B)
        t8=time.time()
        self.grad_hess_time=t8-t7
        return nll, g, h
 
    def _l1GeneralSmooth_sub0(self,gradFunc, w, G, *args):
        options_dict = process_options('verbose', 0, 'threshold', 1e-5,
                                       'optTol', 1e-5, 'maxIter', 500, 'alpha', 5e4,
                                       'update1', 1.25, 'update2', 1.5, 'adjustStep', 1, 'predict', 0)
        verbose, threshold, optTol, maxIter, alpha, update1, update2, adjustStep, predict = \
            [options_dict[key] for key in ['verbose', 'threshold', 'optTol', 'maxIter', 'alpha', 'update1', 'update2', 'adjustStep', 'predict']]
    
        if verbose:
            print('{:10s} {:10s} {:15s} {:15s} {:15s} {:8s} {:15s}'.format(
                'Iteration', 'FunEvals', 'Step Length', 'Function Val', 'Opt Cond', 'Non-Zero', 'Alpha'))
            
        alpha_init = 1
        currParam = alpha_init
        #args = [self.family, self.lmd, X, y]
        f, g, H = gradFunc(w, currParam, G,*args)
        fEvals = 1
        t = 1
        f_prev = f
        t11=time.time()
    
        for i in range(1, maxIter + 1):
            w_old = w
            f_old = f
            t9=time.time()
            d = solveNewton(g, H)
            gtd = np.dot(g.T, d)
            t10=time.time()
            self.time_solvenewton=t10-t9
    
            if gtd > -optTol:
                if verbose:
                    print('Directional Derivative too small')
                break
    
            t, f_prev = initialStepLength(i, adjustStep, f, g, gtd, t, f_prev)
            t, w, f, g, LSfunEvals, H = ArmijoBacktrack(
                w, t, d, f, f, g, gtd, 1e-4, 2, 0, optTol, max(verbose - 1, 0), 0, gradFunc, currParam, G, *args)
            fEvals += LSfunEvals
    
            if verbose:
                print('{:10d} {:10d} {:15.5e} {:15.5e} {:15.5e} {:8d} {:15.5e}'.format(
                    i, fEvals, t, f, np.sum(np.abs(g[np.abs(w) >= threshold])),
                    np.sum(np.abs(w) > threshold), currParam))
    
            oldParam = currParam
            if LSfunEvals == 1:
                currParam = min(currParam * update2, alpha)
            else:
                currParam = min(currParam * update1, alpha)
    
            if verbose == 2 and currParam >= alpha:
                print('At max alpha')
    
            if np.sum(np.abs(g[np.abs(w) >= threshold])) < optTol and oldParam == alpha:
                if verbose:
                    print('Solution Found')
                break
    
            if noProgress(t * d, f, f_old, optTol, verbose):
                break
            elif fEvals > maxIter:
                break
    
            if predict and currParam != oldParam and i % 3 == 0:
                sig = (1 + np.exp(oldParam * w))**-1
                lambda_val = args[1]
                g_alpha = 2 * lambda_val * w * sig * (1 - sig)
    
                predictDir = -np.linalg.solve(H, g_alpha)
                w = w + (currParam - oldParam) * predictDir
                f, g, H = gradFunc(w, currParam, G, *args)
                fEvals += 1
    
        #w[np.abs(w) < threshold] = 0
        t12=time.time()
        self.total_newton=t12-t11
    
        return np.real(w)
    
    def fit(self, X, y):
        X = np.asarray(X,dtype = np.float32)
        y = np.asarray(y,dtype = np.float32)
        if self.fit_intercept:
            X = np.column_stack((np.ones(len(X)), X))        
        beta = np.zeros((X.shape[1], 1))         
        w = self._l1GeneralSmooth_sub0(self._sigmoidL1, beta, self.G, self.family, self.lmd, X, y)
        if self.fit_intercept:
            self.intercept_ = w[0]
            self.coef_ =  w[1:] 
        else:
            self.coef_ = w
    
        return self