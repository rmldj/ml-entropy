# Ising model configurations

* 20000 configurations of the 2D Ising model on a periodic 20x20 lattice generated for the range of temperatures T=1.0 to T=4.0.

* 40000 configurations at T=Tc

The configurations were generated using `code_paper/generate_all_glauber.py` and `code_paper/generate_Tc_glauber.py` in the repository [github.com/rmldj/ml-entropy](https://github.com/rmldj/ml-entropy), which in turn used [github.com/rmldj/ising](https://github.com/rmldj/ising).

## Download data

The generated Monte Carlo configurations can be downloaded from Zenodo (link to appear).
Unpack the downloaded data in this folder.


## Data format

The numpy arrays are in the packed bits format. They should be read in as follows:
```
def unpack(X, lastdim=20):
    return np.unpackbits(X, axis=-1)[:,:,:lastdim] #, count=lastdim)

Xb = unpack(np.load('../data_paper/configurations_{}_glauber.npz'.format(T))['arr_0'])
```

The result is a Nx20x20 `uint8` array. Spins down are 0, spins up are 1. 




