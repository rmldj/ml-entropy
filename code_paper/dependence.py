import numpy as np
import logging
from time import time
import sys
sys.path.append('..')

from mlentropy import entropy_xgb, entropy_lr

logging.basicConfig(filename='dependence.log', format='%(message)s', level=logging.DEBUG)


np.random.seed(0)


n = 10000
nf = 25

permutation = np.random.permutation(3*nf)

X1 = np.random.binomial(1, 0.5, size=(n,nf))
X2 = np.random.binomial(1, 0.5, size=(n,nf))

print(X1.shape, X1.dtype, np.unique(X1, return_counts=True), np.mean(X1))
print(X2.shape, X2.dtype, np.unique(X2, return_counts=True), np.mean(X2))

X3 = X1
X4 = np.logical_or(X1, X2).astype(int)
X5 = np.logical_and(X1, X2).astype(int)
X6 = np.logical_xor(X1, X2).astype(int)

print(X3.shape, X3.dtype, np.unique(X3, return_counts=True), np.mean(X3))
print(X4.shape, X4.dtype, np.unique(X4, return_counts=True), np.mean(X4))
print(X5.shape, X5.dtype, np.unique(X5, return_counts=True), np.mean(X5))
print(X6.shape, X6.dtype, np.unique(X6, return_counts=True), np.mean(X6))
print()

X_copy = np.concatenate((X1, X2, X3), axis=1)[:, permutation]
X_or = np.concatenate((X1, X2, X4), axis=1)[:, permutation]
X_and = np.concatenate((X1, X2, X5), axis=1)[:, permutation]
X_xor = np.concatenate((X1, X2, X6), axis=1)[:, permutation]

np.random.seed(0)
t0 = time()
S_copy_xgb = entropy_xgb(X_copy)
elapsed_min = (time()-t0)/60
logging.info('COPY xgb  {:.2f} time: {:.2f}'.format(S_copy_xgb, elapsed_min))

np.random.seed(0)
t0 = time()
S_copy_lr = entropy_lr(X_copy)
elapsed_min = (time()-t0)/60
logging.info('COPY lr   {:.2f} time: {:.2f}'.format(S_copy_lr, elapsed_min))


np.random.seed(0)
t0 = time()
S_or_xgb = entropy_xgb(X_or)
elapsed_min = (time()-t0)/60
logging.info('OR   xgb  {:.2f} time: {:.2f}'.format(S_or_xgb, elapsed_min))

np.random.seed(0)
t0 = time()
S_or_lr = entropy_lr(X_or)
elapsed_min = (time()-t0)/60
logging.info('OR   lr   {:.2f} time: {:.2f}'.format(S_or_lr, elapsed_min))


np.random.seed(0)
t0 = time()
S_and_xgb = entropy_xgb(X_and)
elapsed_min = (time()-t0)/60
logging.info('AND  xgb  {:.2f} time: {:.2f}'.format(S_and_xgb, elapsed_min))

np.random.seed(0)
t0 = time()
S_and_lr = entropy_lr(X_and)
elapsed_min = (time()-t0)/60
logging.info('AND  lr   {:.2f} time: {:.2f}'.format(S_and_lr, elapsed_min))


np.random.seed(0)
t0 = time()
S_xor_xgb = entropy_xgb(X_xor)
elapsed_min = (time()-t0)/60
logging.info('XOR  xgb  {:.2f} time: {:.2f}'.format(S_xor_xgb, elapsed_min))

np.random.seed(0)
t0 = time()
S_xor_lr = entropy_lr(X_xor)
elapsed_min = (time()-t0)/60
logging.info('XOR  lr   {:.2f} time: {:.2f}'.format(S_xor_lr, elapsed_min))


