import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from xgboost import XGBClassifier



def entropy_xgb(X, n_splits=5, verbose=True, compare=0, compare_method='grassberger', base2=True, eps=1e-8, gpu=False, **kwargs):
    if gpu:
        kwargs['tree_method'] = 'gpu_hist'
        #kwargs.setdefault('n_jobs', 4)
    else:
        kwargs.setdefault('tree_method', 'hist')
        kwargs.setdefault('n_jobs', -1)
    return entropy_ml(X, XGBClassifier, n_splits=n_splits, verbose=verbose, compare=compare, compare_method=compare_method, base2=base2, eps=eps, **kwargs)


def entropy_lr(X, n_splits=5, verbose=True, compare=0, compare_method='grassberger', base2=True, eps=1e-8, **kwargs):
    kwargs.setdefault('solver', 'liblinear')
    return entropy_ml(X, LogisticRegression, n_splits=n_splits, verbose=verbose, compare=compare, compare_method=compare_method, base2=base2, eps=eps, **kwargs)


def entropy_ml(X, ClassifierClass, n_splits=5, verbose=True, compare=0, compare_method='grassberger', base2=True, eps=1e-8, **kwargs):

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
        if compare>0:
            print('{} {:.3f} {:.5f} [{:.5f}]'.format(1, S, S, S))
        else:
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

        S += - np.mean(y*log(p+eps) + (1-y)*log(1-p+eps))

        if verbose:
            if compare>i:
                S_direct = entropy_histogram(X[:,:i+1], method=compare_method)
                print('{} {:.3f} {:.5f} [{:.5f}]'.format(i+1, S, S/(i+1), S_direct/(i+1)))
            else:
                print('{} {:.3f} {:.5f}'.format(i+1, S, S/(i+1)))

    return S

# various entropy estimators based on histograms (the direct one is method='ml')

from scipy.special import digamma

def evenodd(h):
    return 1 - 2*(h%2)

def G(h):
    return digamma(h)+0.5*evenodd(h)*(digamma((h+1)/2) - digamma(h/2))


def get_counts_np(X):
    n = len(X)
    Xb = X.reshape((n,-1))
    p = 2**Xb.shape[1]
    powersof2 = 2**np.arange(Xb.shape[1])
    Xb = np.dot(Xb, powersof2)
    _, counts = np.unique(Xb, return_counts=True)
    return counts, n, p

def entropy_histogram(X, method='ml'):
    counts, n, p = get_counts_np(X)
    thkML = counts/n

    if method=='ml':
        return -np.sum(thkML*np.log2(thkML))

    if method=='chao-shen':
        m1 = np.sum(counts==1)
        thkGT = (1 - m1/n)*thkML
        cf = 1/(1-(1-thkGT)**n)
        return -np.sum(cf*thkGT*np.log2(thkGT))

    if method=='james-stein':
        tk = np.ones_like(counts)/p
        nmiss = p - len(counts)
        #print(p, nmiss/p)
        lm = (1 - np.sum(thkML**2))/((n-1) * (np.sum((tk - thkML)**2) + nmiss/p**2 ))
        thkShrink = lm*tk + (1-lm)*thkML
        Spresent = -np.sum(thkShrink*np.log2(thkShrink))
        Smissing = -nmiss*( lm/p * np.log2(lm/p) )
        return Spresent + Smissing

    if method=='miller-madow':
        K = len(counts)
        return -np.sum(thkML*np.log2(thkML)) +(K-1)/(2*n)/np.log(2)

    if method=='grassberger':
        return (np.log(n) - 1/n * np.sum(counts*G(counts)))/np.log(2)

