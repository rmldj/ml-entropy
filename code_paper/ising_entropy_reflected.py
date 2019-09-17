import numpy as np
import argparse
import logging
from time import time
import sys
sys.path.append('..')

from mlentropy import entropy_xgb, entropy_lr


parser = argparse.ArgumentParser(description='Simulations on CIFAR10')

parser.add_argument('-n', default=20000, type=int, help='use first n samples (default: 20000)')
parser.add_argument('-k', default=5, type=int, help='number of cv folds (default: 5)')
parser.add_argument('--depth', default=3, type=int, help='max depth of trees (default: 3)')
parser.add_argument('--n-estimators', dest='n_estimators', default=100, type=int, help='number of trees (default: 100)')
parser.add_argument('--order', default='natural', type=str, help='ordering: natural|maxcorr|meancorr|SEED (default: natural)')
parser.add_argument('-T', default='Tc', type=str, help='temperature (1.0-4.0 every 0.1 or Tc)')
parser.add_argument('--log', default='ising_entropy', type=str, help='log file (default: ising_entropy)')

args = parser.parse_args()

cmdline = ' '.join(sys.argv)
print(cmdline)
print()
print(args)
print()

import logging
logging.basicConfig(filename='{}.log'.format(args.log), format='%(message)s', level=logging.DEBUG)


def energy_observables(Xb):
    X = 2.0*Xb - 1.0
    N = X.shape[1]*X.shape[2]

    enbulk = np.sum(X[:,1:,:]*X[:,:-1,:], axis=(1,2)) + np.sum(X[:,:,1:]*X[:,:,:-1], axis=(1,2))
    enbndry = np.sum(X[:,0,:]*X[:,-1,:], axis=1) + np.sum(X[:,:,0]*X[:,:,-1], axis=1)
    en = -(enbulk + enbndry)
    return np.mean(en)/N, np.var(en)/N


def correlation_order(X, maxcorr=True):
    assert len(X.shape)==2
    nf = X.shape[1]
    corr = np.abs(np.corrcoef(X, rowvar=False))

    lst = [0]
    rest = list(range(1,nf))
    for i in range(1, nf):
        sumcorr = np.sum( corr[lst,:][:,rest], axis=0 )
        if maxcorr:
            j = np.argmax(sumcorr)
        else:
            j = np.argmin(sumcorr)
        k = rest[j]
        #print(k, sumcorr[j], np.amin(sumcorr), np.amax(sumcorr))
        lst.append(k)
        rest.remove(k)
    return lst


def unpack(X, lastdim=20):
    return np.unpackbits(X, axis=-1)[:,:,:lastdim] #, count=lastdim)

if args.T=='Tc':
    T = 2.2691853
else:
    T = float(args.T)

Xb = unpack(np.load('../data_paper/configurations_{}_glauber.npz'.format(args.T))['arr_0'])

Xb = Xb[:args.n]

energy, var_energy = energy_observables(Xb)

Xb = Xb.reshape((len(Xb),-1))
nf = Xb.shape[1]



if args.order=='natural':
    pass
elif args.order=='maxcorr':
    ordering = correlation_order(Xb)
    Xb = Xb[:, ordering]
elif args.order=='mincorr':
    ordering = correlation_order(Xb, maxcorr=False)
    Xb = Xb[:, ordering]
else:
    SEED = int(args.order)
    np.random.seed(SEED)
    Xb = Xb[:, np.random.permutation(nf)]


Xbreflected = 1-Xb
Xb = np.concatenate((Xb, Xbreflected), axis=0)

np.random.seed(0)

t0 = time()
S = entropy_xgb(Xb, n_splits=args.k, compare=10, n_estimators=args.n_estimators, max_depth=args.depth)
elapsed_min = (time()-t0)/60

S = S/nf
F = energy - T*S*np.log(2)
logging.info(cmdline+' | {} {} {:.5f} {:.5f} {:.5f} {:.5f} {:.2f}'.format(args.T, T, energy, var_energy, F, S, elapsed_min))
