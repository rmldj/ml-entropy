import numpy as np
import sys
sys.path.append('..')

from mlentropy import xgbentropy

T = sys.argv[1]
print('temperature', T)

def unpack(X, lastdim=20):
    return np.unpackbits(X, axis=-1)[:,:,:lastdim] #, count=lastdim)

Xb = unpack(np.load('../data_paper/configurations_{}.npz'.format(T))['arr_0'])

Xb = Xb[:10000]

X = 2.0*Xb - 1.0

enbulk = np.sum(X[:,1:,:]*X[:,:-1,:], axis=(1,2)) + np.sum(X[:,:,1:]*X[:,:,:-1], axis=(1,2))
enbndry = np.sum(X[:,0,:]*X[:,-1,:], axis=1) + np.sum(X[:,:,0]*X[:,:,-1], axis=1)
en = -(enbulk + enbndry)

N = 20**2

print(Xb.shape, Xb.dtype)
print(X.shape, X.dtype)
print()
print('energy', np.mean(en)/N, np.median(en)/N, 'std', np.std(en)/N, 'var', np.var(en)/N)
print()


S = xgbentropy(Xb, gpu=False)
print()
print('entropy', S/N)
