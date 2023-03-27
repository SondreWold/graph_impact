
for seed in 42 1443 1790 1234 5432 654 3 222 90 12
do
    python3 train.py --seed=$seed --model="bert-base-uncased" --epochs=10 --task="expla" --batch_size=8 --lr=3e-5 --weight_decay=0.01 --test
    python3 train.py --seed=$seed --model="bert-base-uncased" --epochs=10 --task="expla" --batch_size=8 --use_graphs --lr=5e-5 --weight_decay=0.1 --test
    python3 train.py --seed=$seed --model="bert-base-uncased" --epochs=10 --task="expla" --batch_size=8 --use_graphs --lr=5e-5 --weight_decay=0.1 --test --el
    python3 train.py --seed=$seed --model="bert-base-uncased" --epochs=10 --task="expla" --batch_size=8 --use_graphs --lr=5e-5 --weight_decay=0.1 --test --rg
    #python3 train.py --seed=$seed --model="bert-base-uncased" --epochs=10 --task="expla" --batch_size=8 --use_graphs --lr=5e-5 --weight_decay=0.1 --test --pg

    python3 train.py --seed=$seed --model="bert-base-uncased" --epochs=10 --task="copa" --batch_size=4 --lr=6e-6 --weight_decay=0.01 --test
    python3 train.py --seed=$seed --model="bert-base-uncased" --epochs=10 --task="copa" --batch_size=8 --use_graphs --lr=3e-5 --weight_decay=0.1 --test
    python3 train.py --seed=$seed --model="bert-base-uncased" --epochs=10 --task="copa" --batch_size=8 --use_graphs --lr=3e-5 --weight_decay=0.1 --test --el
    python3 train.py --seed=$seed --model="bert-base-uncased" --epochs=10 --task="copa" --batch_size=8 --use_graphs --lr=3e-5 --weight_decay=0.1 --test --rg
    #python3 train.py --seed=$seed --model="bert-base-uncased" --epochs=10 --task="copa" --batch_size=8 --use_graphs --lr=3e-5 --weight_decay=0.1 --test --pg
done