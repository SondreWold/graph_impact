
for seed in 9 117269 76230 41480 60050
do
    python3 train.py --seed=$seed --dropout=0.3 --model="bert-base-uncased" --epochs=10 --task="expla" --batch_size=8 --lr=4e-5 --weight_decay=0.1 --test
    python3 train.py --seed=$seed --dropout=0.3 --model="bert-base-uncased" --epochs=10 --task="expla" --batch_size=16 --use_graphs --lr=3e-5 --weight_decay=0.01 --test
    python3 train.py --seed=$seed --dropout=0.3 --model="bert-base-uncased" --epochs=10 --task="expla" --batch_size=16 --use_graphs --lr=3e-5 --weight_decay=0.01 --test --el
    python3 train.py --seed=$seed --dropout=0.3 --model="bert-base-uncased" --epochs=10 --task="expla" --batch_size=16 --use_graphs --lr=3e-5 --weight_decay=0.01 --test --rg
    python3 train.py --seed=$seed --dropout=0.3 --model="bert-base-uncased" --epochs=10 --task="expla" --batch_size=16 --use_graphs --lr=3e-5 --weight_decay=0.01 --test --pgg
    python3 train.py --seed=$seed --dropout=0.3 --model="bert-base-uncased" --epochs=10 --task="expla" --batch_size=16 --use_graphs --lr=3e-5 --weight_decay=0.01 --test --pgl

    python3 train.py --seed=$seed --dropout=0.3 --model="bert-large-uncased" --epochs=10 --task="copa" --batch_size=16 --lr=4e-6 --weight_decay=0.1 --test
    python3 train.py --seed=$seed --dropout=0.3 --model="bert-large-uncased" --epochs=10 --task="copa" --batch_size=8 --use_graphs --lr=4e-5 --weight_decay=0.01 --test
    python3 train.py --seed=$seed --dropout=0.3 --model="bert-large-uncased" --epochs=10 --task="copa" --batch_size=8 --use_graphs --lr=4e-5 --weight_decay=0.01 --test --el
    python3 train.py --seed=$seed --dropout=0.3 --model="bert-large-uncased" --epochs=10 --task="copa" --batch_size=8 --use_graphs --lr=4e-5 --weight_decay=0.01 --test --rg
    python3 train.py --seed=$seed --dropout=0.3 --model="bert-large-uncased" --epochs=10 --task="copa" --batch_size=8 --use_graphs --lr=4e-5 --weight_decay=0.01 --test --pgg
    python3 train.py --seed=$seed --dropout=0.3 --model="bert-large-uncased" --epochs=10 --task="copa" --batch_size=8 --use_graphs --lr=4e-5 --weight_decay=0.01 --test --pl

done