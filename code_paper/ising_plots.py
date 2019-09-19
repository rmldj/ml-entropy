import numpy as np
import matplotlib.pyplot as plt

Ts = np.arange(1.0,4.1,0.1)


entropies20 = np.loadtxt('entropies20.txt')
free20 = np.loadtxt('freeenergies20.txt')
energies20 = np.loadtxt('energies20.txt')
varenergies20 = np.loadtxt('varenergies20.txt')

n = len(Ts)

entropy = np.zeros(n)
free = np.zeros(n)
energy = np.zeros(n)
varenergy = np.zeros(n)

with open('scan50.log') as f:
    for i, ln in enumerate(f):
        _, data = ln.strip().split('|')
        data = [float(s) for s in data.split()]
        free[i] = data[4]
        entropy[i] = data[5]
        energy[i] = data[2]
        varenergy[i] = data[3]
        print(data)

entropyaug = np.zeros(n)
freeaug = np.zeros(n)

with open('scan50augment.log') as f:
    for i, ln in enumerate(f):
        _, data = ln.strip().split('|')
        data = [float(s) for s in data.split()]
        freeaug[i] = data[4]
        entropyaug[i] = data[5]


plt.figure()
plt.plot(Ts, entropies20, 'k', label='exact L=20 2D Ising')
plt.plot(Ts, entropyaug, 'mo', label='xgboost entropy augmented')
plt.plot(Ts, entropy, 'ro', label='xgboost entropy')
plt.xlabel('T')
plt.ylabel('S')
plt.legend()

plt.figure()
plt.plot(Ts, energies20, 'k', label='exact L=20 2D Ising')
plt.plot(Ts, energy, 'ro', label='MC energy')
plt.xlabel('T')
plt.ylabel('E')
plt.legend()

plt.figure()
plt.plot(Ts, varenergies20, 'k', label='exact L=20 2D Ising')
plt.plot(Ts, varenergy, 'ro', label='MC energy variance')
plt.xlabel('T')
plt.ylabel('var E')
plt.legend()

plt.figure()
plt.plot(Ts, free20, 'k', label='exact L=20 2D Ising')
plt.plot(Ts, freeaug, 'mo', label='xgboost entropy augmented + MC energy')
plt.plot(Ts, free, 'ro', label='xgboost entropy + MC energy')
plt.xlabel('T')
plt.ylabel('F')
plt.legend()


plt.show()
