


#Batch size 16

python train.py --seed=1234 --model="roberta-base" --use_graph --epochs=4 --lr=3e-5 --pg 
python train.py --seed=1234 --model="roberta-base" --use_graph --epochs=4 --lr=3e-5
python train.py --seed=1234 --model="roberta-base" --epochs=4 --lr=3e-5

python train.py --seed=1234 --model="roberta-base" --use_graph --epochs=4 --lr=4e-5 --pg
python train.py --seed=1234 --model="roberta-base" --use_graph --epochs=4 --lr=4e-5
python train.py --seed=1234 --model="roberta-base" --epochs=4 --lr=4e-5

python train.py --seed=1234 --model="roberta-base" --use_graph --epochs=4 --lr=5e-5 --pg
python train.py --seed=1234 --model="roberta-base" --use_graph --epochs=4 --lr=5e-5
python train.py --seed=1234 --model="roberta-base" --epochs=4 --lr=5e-5


#Batch size 32

python train.py --seed=1234 --model="roberta-base" --use_graph --epochs=4 --lr=3e-5 --pg --batch_size=32
python train.py --seed=1234 --model="roberta-base" --use_graph --epochs=4 --lr=3e-5 --batch_size=32
python train.py --seed=1234 --model="roberta-base" --epochs=4 --lr=3e-5 --batch_size=32

python train.py --seed=1234 --model="roberta-base" --use_graph --epochs=4 --lr=4e-5 --pg --batch_size=32
python train.py --seed=1234 --model="roberta-base" --use_graph --epochs=4 --lr=4e-5 --batch_size=32
python train.py --seed=1234 --model="roberta-base" --epochs=4 --lr=4e-5 --batch_size=32

python train.py --seed=1234 --model="roberta-base" --use_graph --epochs=4 --lr=5e-5 --pg --batch_size=32
python train.py --seed=1234 --model="roberta-base" --use_graph --epochs=4 --lr=5e-5 --batch_size=32
python train.py --seed=1234 --model="roberta-base" --epochs=4 --lr=5e-5 --batch_size=32


