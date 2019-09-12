import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from xgboost import XGBClassifier


def xgbentropy(X, n_splits=5, verbose=True, base2=True, eps=1e-8, gpu=True, **kwargs):
    if gpu:
        kwargs['tree_method'] = 'gpu_hist'
        kwargs.setdefault('n_jobs', 4)
    else:
        kwargs.setdefault('tree_method', 'hist')
        kwargs.setdefault('n_jobs', -1)
    return mlentropy(X, XGBClassifier, n_splits=n_splits, verbose=verbose, base2=base2, eps=eps, **kwargs)


def lrentropy(X, n_splits=5, verbose=True, base2=True, eps=1e-8, **kwargs):
    return mlentropy(X, LogisticRegression, n_splits=n_splits, verbose=verbose, base2=base2, eps=eps, **kwargs)


def mlentropy(X, ClassifierClass, n_splits=5, verbose=True, base2=True, eps=1e-8, **kwargs):

    if base2:
        log = np.log2
    else:
        log = np.log

    if(len(X.shape)>2):
        X = X.reshape((len(X), -1))
    nf = X.shape[1]
    
    p = np.mean(X[:,0])
    S = -p*log(p+eps) - (1-p)*log(1-p+eps)

    if verbose:
        print('{} {:.3f} {:.5f}'.format(1, S, S))

    for i in range(1,nf):
        XX = X[:,:i]
        y = X[:,i]
        p = np.zeros(len(y))

        kf = KFold(n_splits=n_splits, shuffle=True)
        # the splits are different for each fitted feature
        # for reproducibility set np.random.seed(SEED) before calling mlentropy/xgbentropy etc.

        for tr, tst in kf.split(XX):
            clf = ClassifierClass(**kwargs)
            clf.fit(XX[tr],y[tr])
            p[tst] = clf.predict_proba(XX[tst])[:,1]

        S += - np.mean(y*log(p+eps) + (1-y)*log(1-p)+eps)

        if verbose:
            print('{} {:.3f} {:.5f}'.format(i+1, S, S/(i+1)))

    return S

