python -u full-supervised.py --data cora --layer 64 --alpha 0.2 --weight_decay 1e-4
python -u full-supervised.py --data cora --layer 64 --alpha 0.2 --weight_decay 1e-4 --variant
python -u full-supervised.py --data citeseer --layer 64 --weight_decay 5e-6
python -u full-supervised.py --data citeseer --layer 64 --weight_decay 5e-6 --variant
python -u full-supervised.py --data pubmed --layer 64 --alpha 0.1 --weight_decay 5e-6
python -u full-supervised.py --data pubmed --layer 64 --alpha 0.1 --weight_decay 5e-6 --variant
python -u full-supervised.py --data chameleon --layer 8 --lamda 1.5 --alpha 0.2 --weight_decay 5e-4
python -u full-supervised.py --data chameleon --layer 8 --lamda 1.5 --alpha 0.2 --weight_decay 5e-4 --variant
python -u full-supervised.py --data cornell --layer 16 --lamda 1 --weight_decay 1e-3
python -u full-supervised.py --data cornell --layer 16 --lamda 1 --weight_decay 1e-3 --variant
python -u full-supervised.py --data texas --layer 32 --lamda 1.5 --weight_decay 1e-4
python -u full-supervised.py --data texas --layer 32 --lamda 1.5 --weight_decay 1e-4 --variant
python -u full-supervised.py --data wisconsin --layer 16 --lamda 1 --weight_decay 5e-4
python -u full-supervised.py --data wisconsin --layer 16 --lamda 1 --weight_decay 5e-4 --variant