import numpy as np
import os

Ts = np.arange(1.0,4.1,0.1)
Tss = ['{:.1f}'.format(T) for T in Ts]

for T in Tss:
    cmd = '~/Programming/ising/install/bin/ising -d 2 -L 20 -T {0} --nmeas 1000 --nmcs 20000000 --ieq 5000 --dyn 3 --print-state'.format(T)
    outfname = '../data_paper/configurations_wolff/configurations_{}.npz'.format(T)
    print(cmd)
    os.system('rm estats/*')
    os.system(cmd)
    
    X = np.zeros((20000, 20, 20), dtype=np.uint8)
    for i in range(20000):
        snapshot = np.loadtxt('estats/estat{}.txt'.format(i+1), dtype=int)
        X[i][snapshot>0] = 1 
    print(outfname, np.mean(X))
    np.savez_compressed(outfname, np.packbits(X, axis=-1))

os.system('rm estats/*')
