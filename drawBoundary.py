from numpy import *
from pylab import *

class Boundary:
    
    def __init__(self, dname):
        self.dataName = dname
        self.modelName = dname + ".model"

    def plotData(self,X,Y):
        plot(X[Y>=0,0], X[Y>=0,1], 'bs', markersize=5)
        plot(X[Y< 0,0], X[Y< 0,1], 'ro', markersize=6)  
    
    def readData(self,filename):
        h = open(filename, 'r')
        Y = []
        X = []
        for l in h.readlines():
            a = l.strip().split()
            y = float(a[0])
            x1 = float(a[1][2:])
            x2 = float(a[2][2:])
            Y.append(y)
            X.append(array([x1,x2]))
        h.close()
        return (array(X),array(Y))

    def loadLibSVMModel(self,filename):
        h = open(filename, 'r')
        params = {}
        svs = []
        alpha = []
        inHeader = True
        for l in h.readlines():
            l = l.strip()
            if l == "SV":
                inHeader = False
            elif inHeader:
                a = l.split()
                params[a[0]] = a[1:]
            else:
                a = l.split()
                y = float(a[0])
                x1 = float(a[1][2:])
                x2 = float(a[2][2:])
                alpha.append(y)
                svs.append(array([x1,x2]))
        h.close()
        return (params, array(alpha), array(svs))

    def plotSVS(self,alpha, svs):
        plot(svs[alpha>0,0], svs[alpha>0,1], 'bs', markersize=10)
        plot(svs[alpha<0,0], svs[alpha<0,1], 'ro', markersize=11)

    def computeKernel(self,params, alpha, svs, D):
        if params['kernel_type'][0] == 'linear':
            rho    = float(params['rho'][0])
            return sum(alpha * dot(D, svs.T), axis=1) - rho
        if params['kernel_type'][0][:4] == 'poly':
            degree = float(params['degree'][0])
            gamma  = float(params['gamma'][0])
            coef0  = float(params['coef0'][0])
            rho    = float(params['rho'][0])
            # (gamma*u'*v + coef0)^degree
            return sum(alpha * ((gamma * dot(D, svs.T) + coef0) ** degree), axis=1) - rho
        if params['kernel_type'][0] == 'rbf':
            gamma  = float(params['gamma'][0])
            rho    = float(params['rho'][0])
            # exp(-gamma*|u-v|^2)
            N = D.shape[0]
            Z = zeros((N,)) - rho
            for i in range(alpha.shape[0]):
                v = repeat(reshape(svs[i,:],(1,2)),N,axis=0) - D
                Z = Z + alpha[i] * exp(-gamma*sum(v*v,axis=1))
            return Z
        if params['kernel_type'][0][:3] == 'sig':
            gamma  = float(params['gamma'][0])
            coef0  = float(params['coef0'][0])
            rho    = float(params['rho'][0])
            # tanh(gamma*u'*v + coef0)
            return sum(alpha * (tanh(gamma * dot(D, svs.T) + coef0)), axis=1) - rho
        raise Exception('unknown kernel type: ' + params['kernel_type'][0])



    def plotContour(self, params, alpha, svs, resolution=0.005):
        n = len(arange(0,1,resolution))
        X0 = arange(0,1,resolution) * ones((n,n)) - 0.5
        Y0 = X0.T
        D = array([X0.reshape(-1), Y0.reshape(-1)]).T
        K = self.computeKernel(params, alpha, svs, D)
        Z = K.reshape((n,n))
        numCont = 100
        colors = []
        half = int(numCont/2)
        for i in range(numCont):
            r = 0
            b = 0
            if i < half:
                b = 0.25 + (half - i) / float(half) * 2/4
            else:
                r = 0.25 + (i - half) / float(half) * 2/4
            colors.append((r,0,b))
        Zmax = abs(Z).max()
        levels = arange(-Zmax,Zmax,Zmax/50)
        contourf(X0, Y0, -Z, levels=levels,colors=colors)
        contour(X0, Y0, Z, levels=[0], linewidths=[5], colors='w')
        contour(X0, Y0, Z, levels=[-1], linewidths=[2], colors='w', linestyles='dashed')
        contour(X0, Y0, Z, levels=[1], linewidths=[2], colors='w', linestyles='dashed')
        return Z

    def plotAll(self,X, Y, params, alpha, svs):
        figure(1)
        #hold(True)
        print(params)
        print(alpha)
        print(svs)
        Z = self.plotContour(params, alpha, svs)
        
        self.plotData(X,Y)
        self.plotSVS(alpha, svs)
        show()
        return Z
    
    def draw(self):
        (X,Y) = self.readData(self.dataName)
        (params,alpha,svs) = self.loadLibSVMModel(self.modelName)
        Z = self.plotAll(X,Y,params,alpha,svs)