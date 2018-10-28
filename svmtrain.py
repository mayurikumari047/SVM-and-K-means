from sklearn.svm import SVC
import numpy as np
import optparse
import sys

class SVMWrapper:
    def __init__(self, args):
        if len(args) < 2: raise Exception()
        self.c = 1.0
        self.k = 'rbf'
        self.d = 3
        self.g = 1.0
        self.r = 0.0
        self.filename = args[-2]
        self.model = args[-1]
        print("training model {0} -> {1}".format(self.filename,self.model))
        i = 0
        while i < len(args) - 2:
            if args[i] == '-t':
                self.k = self.string2kernel(args[i+1])
                i += 2
            elif args[i] == '-c':
                self.c = float(args[i+1])
                i += 2
            elif args[i] == '-r':
                self.r = float(args[i+1])
                i += 2
            elif args[i] == '-g':
                self.g = float(args[i+1])
                i += 2
            elif args[i] == '-d':
                self.d = float(args[i+1])
                i += 2
            else:
                raise Exception('invalid argument: ' + str(args[i]))

    def loadData(self):
        X = []
        Y = []
        with open(self.filename, 'r') as h:
            for l in h.readlines():
                a = l.strip().split()
                # assumes all features are defined!
                Y.append(float(a[0]))
                # parse the features
                X.append([float(i.split(':', 1)[1]) for i in a[1:]])
        return np.array(X),np.array(Y)

    def string2kernel(self,s):
        if s == '0': return 'linear'
        elif s == '1': return 'poly'
        elif s == '2': return 'rbf'
        elif s == '3': return 'tanh'
        else: return s

    def train(self):
        X,Y = self.loadData()
        f = SVC(C=self.c, kernel=self.k, degree=self.d, gamma=self.g, coef0=self.r)
        f.fit(X,Y)
        print("fit model:{0}".format(f), file=sys.stderr)
        with open(self.model, 'w') as h:
            possv,negsv = 0,0
            #if f.classes_[0] == -1:
            #    f.dual_coef_ *= -1
            
            for alpha in f.dual_coef_[0]:
                if alpha < 0: negsv += 1
                else: possv += 1
                    
            print("svm_type c_svc", file=h)
            print("kernel_type {0}".format(self.k), file=h)
            print("gamma {0}".format(self.g), file=h)
            print("degree {0}".format(self.d), file=h)
            print("coef0 {0}".format(self.r), file=h)
            print("nr_class {0}".format(2), file=h)
            print("total_sv {0}".format(np.sum(f.n_support_)), file=h) 
            print("rho {0}".format(-f.intercept_[0]), file=h) 
            print("label -1 1",file=h)
            print("nr_sv {0} {1}".format(possv, negsv), file=h) 
            print("SV", file=h)
            
            for i,alpha in enumerate(f.dual_coef_[0]):
                if alpha < 0: continue
                print(str(alpha) + ' ' + ' '.join([str(fn+1)+':'+str(fv) for fn,fv in enumerate(f.support_vectors_[i])]), file=h)
                
            for i,alpha in enumerate(f.dual_coef_[0]):
                if alpha > 0: continue
                print(str(alpha) + ' ' + ' '.join([str(fn+1)+':'+str(fv) for fn,fv in enumerate(f.support_vectors_[i])]), file=h)

    if __name__ == '__main__' and len(sys.argv) > 0 and len(sys.argv[0]) > 0:
        main(sys.argv[1:])

