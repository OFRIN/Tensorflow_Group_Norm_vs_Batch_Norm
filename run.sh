python Train.py --batch_size=1 --norm_type=batch
python Train.py --batch_size=2 --norm_type=batch
python Train.py --batch_size=4 --norm_type=batch
python Train.py --batch_size=8 --norm_type=batch
python Train.py --batch_size=16 --norm_type=batch
python Train.py --batch_size=32 --norm_type=batch
python Train.py --batch_size=64 --norm_type=batch

python Train.py --batch_size=1 --norm_type=group
python Train.py --batch_size=2 --norm_type=group
python Train.py --batch_size=4 --norm_type=group
python Train.py --batch_size=8 --norm_type=group
python Train.py --batch_size=16 --norm_type=group
python Train.py --batch_size=32 --norm_type=group
python Train.py --batch_size=64 --norm_type=group

python Generate_Graph.py
