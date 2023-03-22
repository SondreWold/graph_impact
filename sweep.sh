
for lr in 1e-5 3e-5 5e-5 1e-4 5e-4 8e-6 6e-6 4e-6 2e-6 1e-6
do
    for batch_size in 4 8 16 32 64
    do
        for weight_decay in 0.1 0.01
        do
            python3 train.py --model="bert-base-uncased" --epochs=10 --task="expla" --batch_size=$batch_size --lr=$lr --weight_decay=$weight_decay
        done
    done
done


for lr in 1e-5 3e-5 5e-5 1e-4 5e-4 8e-6 6e-6 4e-6 2e-6 1e-6
do
    for batch_size in 4 8 16 32 64
    do
        for weight_decay in 0.1 0.01
        do
            python3 train.py --model="bert-base-uncased" --epochs=10 --task="copa" --batch_size=$batch_size --lr=$lr --weight_decay=$weight_decay
        done
    done
done


for lr in 1e-5 3e-5 5e-5 1e-4 5e-4 8e-6 6e-6 4e-6 2e-6 1e-6
do
    for batch_size in 4 8 16 32 64
    do
        for weight_decay in 0.1 0.01
        do
            python3 train.py --model="bert-base-uncased" --epochs=10 --task="expla" --use_graphs --batch_size=$batch_size --lr=$lr --weight_decay=$weight_decay
        done
    done
done


for lr in 1e-5 3e-5 5e-5 1e-4 5e-4 8e-6 6e-6 4e-6 2e-6 1e-6
do
    for batch_size in 4 8 16 32 64
    do
        for weight_decay in 0.1 0.01
        do
            python3 train.py --model="bert-base-uncased" --epochs=10 --task="copa" --use_graphs --batch_size=$batch_size --lr=$lr --weight_decay=$weight_decay
        done
    done
done