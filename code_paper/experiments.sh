python ising_entropy.py --log order --order natural
python ising_entropy.py --log order --order maxcorr
python ising_entropy.py --log order --order mincorr
python ising_entropy.py --log order --order 0
python ising_entropy.py --log order --order 137
python ising_entropy.py --log order --order 555
python ising_entropy.py --log order --order 1621
python ising_entropy.py --log order --order 4567


python ising_entropy.py --log n -n 10000
python ising_entropy.py --log n -n 20000
python ising_entropy.py --log n -n 40000


python ising_entropy.py --log cv -k 2
python ising_entropy.py --log cv -k 3
python ising_entropy.py --log cv -k 4
python ising_entropy.py --log cv -k 5


python ising_entropy.py --log hyper --depth 3 --n-estimators 100
python ising_entropy.py --log hyper --depth 3 --n-estimators 200
python ising_entropy.py --log hyper --depth 5 --n-estimators 100
python ising_entropy.py --log hyper --depth 5 --n-estimators 200
