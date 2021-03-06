import numpy as np
import os



T = '2.2691853'

cmd = '~/Programming/ising/install/bin/ising -d 2 -L 20 -T {0} --nmeas 1000 --nmcs 40000000 --ieq 5000 --dyn 0 --print-state'.format(T)
outfname = '../data_paper/configurations_Tc_glauber.npz'
print(cmd)
os.system(cmd)
    
X = np.zeros((40000, 20, 20), dtype=np.uint8)
for i in range(40000):
    snapshot = np.loadtxt('estats/estat{}.txt'.format(i+1), dtype=int)
    X[i][snapshot>0] = 1 
print(outfname, np.mean(X))
np.savez_compressed(outfname, np.packbits(X, axis=-1))
os.system('rm estats/*')
