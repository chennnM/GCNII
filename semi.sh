python -u train.py --data cora --layer 64 --test
python -u train.py --data cora --layer 64 --variant --test
python -u train.py --data citeseer --layer 32 --hidden 256 --lamda 0.6 --dropout 0.7 --test
python -u train.py --data citeseer --layer 32 --hidden 256 --lamda 0.6 --dropout 0.7 --variant --test
python -u train.py --data pubmed --layer 16 --hidden 256 --lamda 0.4 --dropout 0.5 --wd1 5e-4 --test
python -u train.py --data pubmed --layer 16 --hidden 256 --lamda 0.4 --dropout 0.5 --wd1 5e-4 --variant --test