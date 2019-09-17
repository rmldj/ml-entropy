import numpy as np
import os

Ts = np.arange(1.0,4.1,0.1)
Tss = ['{:.1f}'.format(T) for T in Ts]

for T in Tss:
    cmd = 'python ising_entropy_reflected.py --order 0 --log scan50reflected -T {} --n-estimators 50'.format(T)
    print(cmd)
    os.system(cmd)


