# ml-entropy

Code for the paper 

### Romuald A. Janik, *Entropy from Machine Learning*, [arXiv:1909.10831](https://arxiv.org/abs/1909.10831)

as well as a routine implementing the proposed method of computing entropy. Please cite the paper if you would use this method in a scientific publication.

**UPDATE:** The current version of `mlentropy.py` implements improvements for handling cases with extremely unbalanced data. The results of the paper where computed using the earlier version (choose tag 1.0).


See the README file in the `code_paper/` folder for specific instructions for reproducing the results of the paper

## `mlentropy.py`

This file implements the basic procedures proposed in the paper for computing the entropy. It requires `xgboost`, `scikit-learn`, `scipy` and `numpy`.

The main functions are
```
def entropy_xgb(X, n_splits=5, verbose=True, compare=0, compare_method='grassberger', base2=True, eps=1e-8, gpu=False, **kwargs):
```

for computing the entropy using **xgboost**.

```
def entropy_lr(X, n_splits=5, verbose=True, compare=0, compare_method='grassberger', base2=True, eps=1e-8, **kwargs)
```
which uses logistic regression and
```
def entropy_ml(X, ClassifierClass, n_splits=5, verbose=True, compare=0, compare_method='grassberger', base2=True, eps=1e-8, **kwargs)
```
which uses a generic classifier class with an **scikit-learn** like API.

The parameters are as follows:
* `n_splits` is the number of cross validation folds
* `verbose=True` prints out after each classification problem the feature number, total entropy, entropy per spin/neuron
* `compare=n` in addition prints out entropy per spin computed directly from histograms for the first `n` subsets
* `compare_method='grassberger'` uses a specific variant of entropy estimation from a histogram
* `base2=True` returns entropy in bits
* `eps=1e-8` constant to add to the argument of logarithms to avoid *NaN* when probability is 0 or 1
* `gpu=False` (only for **xgboost**) whether to use GPU

The remaining keyword arguments are directly passed to the constructor of the classifier.
The functions return the total entropy of `X`.

Note that for reproducibility one has to set `np.random.seed(...)` before calling any of these functions. The cross validation involves shuffling the dataset, but `random_state` is not set in order for the cross validation split to be different for each auxiliary classification problem.


In addition 
```
def entropy_histogram(X, method='ml', base2=True):
```
contains routines for estimating entropy from histograms (occurence counts) using `ml` (maximal likelihood), `chao-shen`,
`james-stein`, `miller-madow` and `grassberger` methods. Note that these have not been thoroughly tested.
