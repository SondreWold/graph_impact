
for seed in 9 119 7230 4180 6050 257 981 1088 416 88
do
    python3 train.py --seed=$seed --patience=4 --dropout=0.2 --model="bert-base-uncased" --epochs=10 --task="copa" --batch_size=16 --lr=5e-5 --weight_decay=0.01 --test
    python3 train.py --seed=$seed --patience=4 --dropout=0.3 --model="bert-base-uncased" --epochs=10 --task="copa" --batch_size=4 --use_graphs --lr=4e-5 --weight_decay=0.01 --test
    python3 train.py --seed=$seed --patience=4 --dropout=0.3 --model="bert-base-uncased" --epochs=10 --task="copa" --batch_size=4 --use_graphs --lr=4e-5 --weight_decay=0.01 --test --el
    python3 train.py --seed=$seed --patience=4 --dropout=0.3 --model="bert-base-uncased" --epochs=10 --task="copa" --batch_size=4 --use_graphs --lr=4e-5 --weight_decay=0.01 --test --rg
    python3 train.py --seed=$seed --patience=4 --dropout=0.3 --model="bert-base-uncased" --epochs=10 --task="copa" --batch_size=4 --use_graphs --lr=4e-5 --weight_decay=0.01 --test --pgg
    python3 train.py --seed=$seed --patience=4 --dropout=0.3 --model="bert-base-uncased" --epochs=10 --task="copa" --batch_size=4 --use_graphs --lr=4e-5 --weight_decay=0.01 --test --pl
done