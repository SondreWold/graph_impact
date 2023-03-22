for seed in 123 43 132
do
    for lr in 1e-5 2e-5 3e-5 4e-5 5e-5 1e-6 5e-4
    do
        for batch_size in 8 16 32
        do
            python3 train.py --seed=$seed --model="bert-base-uncased" --epochs=10 --task="copa" --batch_size=$batch_size --lr=$lr
        done
    done
done

for seed in 123 43 132
do
    for lr in 1e-5 2e-5 3e-5 4e-5 5e-5 1e-6 5e-4
    do
        for batch_size in 8 16 32
        do
            python3 train.py --seed=$seed --model="bert-base-uncased" --epochs=10 --task="expla" --batch_size=$batch_size --lr=$lr
        done
    done
done

for seed in 123 43 132
do
    for lr in 1e-5 2e-5 3e-5 4e-5 5e-5 1e-6 5e-4
    do
        for batch_size in 8 16 32
        do
            python3 train.py --seed=$seed --model="bert-base-uncased" --use_graphs --epochs=10 --task="copa" --batch_size=$batch_size --lr=$lr
        done
    done
done

for seed in 123 43 132
do
    for lr in 1e-5 2e-5 3e-5 4e-5 5e-5 1e-6 5e-4
    do
        for batch_size in 8 16 32
        do
            python3 train.py --seed=$seed --model="bert-base-uncased" --use_graphs --epochs=10 --task="expla" --batch_size=$batch_size --lr=$lr
        done
    done
done
