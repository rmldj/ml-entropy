python ising_entropy.py --order 0 --log hyper --depth 3 --n-estimators 100
python ising_entropy.py --order 0 --log hyper --depth 3 --n-estimators 200
python ising_entropy.py --order 0 --log hyper --depth 5 --n-estimators 100
python ising_entropy.py --order 0 --log hyper --depth 5 --n-estimators 200
python ising_entropy.py --order 0 --log hyper --depth 3 --n-estimators 50
python ising_entropy.py --order 0 --log hyper --depth 3 --n-estimators 25
python ising_entropy.py --order 0 --log hyper --depth 3 --n-estimators 75

python ising_entropy.py --order 0 --log n --n-estimators 50 -n 10000
python ising_entropy.py --order 0 --log n --n-estimators 50 -n 20000
python ising_entropy.py --order 0 --log n --n-estimators 50 -n 40000

python ising_entropy.py --order 0 --log cv --n-estimators 50 -k 2
python ising_entropy.py --order 0 --log cv --n-estimators 50 -k 3
python ising_entropy.py --order 0 --log cv --n-estimators 50 -k 4
python ising_entropy.py --order 0 --log cv --n-estimators 50 -k 5


python ising_entropy.py --log order --n-estimators 50 --order natural
python ising_entropy.py --log order --n-estimators 50 --order maxcorr
python ising_entropy.py --log order --n-estimators 50 --order mincorr
python ising_entropy.py --log order --n-estimators 50 --order 0
python ising_entropy.py --log order --n-estimators 50 --order 137
python ising_entropy.py --log order --n-estimators 50 --order 555
python ising_entropy.py --log order --n-estimators 50 --order 1621
python ising_entropy.py --log order --n-estimators 50 --order 4567

#python ising_entropy.py -T 1.5 --log order --n-estimators 50 --order natural
#python ising_entropy.py -T 1.5 --log order --n-estimators 50 --order maxcorr
#python ising_entropy.py -T 1.5 --log order --n-estimators 50 --order mincorr
#python ising_entropy.py -T 1.5 --log order --n-estimators 50 --order 0
#python ising_entropy.py -T 1.5 --log order --n-estimators 50 --order 137
#python ising_entropy.py -T 1.5 --log order --n-estimators 50 --order 555
#python ising_entropy.py -T 1.5 --log order --n-estimators 50 --order 1621
#python ising_entropy.py -T 1.5 --log order --n-estimators 50 --order 4567

#python ising_entropy.py -T 3.5 --log order --n-estimators 50 --order natural
#python ising_entropy.py -T 3.5 --log order --n-estimators 50 --order maxcorr
#python ising_entropy.py -T 3.5 --log order --n-estimators 50 --order mincorr
#python ising_entropy.py -T 3.5 --log order --n-estimators 50 --order 0
#python ising_entropy.py -T 3.5 --log order --n-estimators 50 --order 137
#python ising_entropy.py -T 3.5 --log order --n-estimators 50 --order 555
#python ising_entropy.py -T 3.5 --log order --n-estimators 50 --order 1621
#python ising_entropy.py -T 3.5 --log order --n-estimators 50 --order 4567






