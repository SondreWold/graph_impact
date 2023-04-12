
for lr in 4e-5 3e-5 5e-5 6e-6 4e-6 1e-6
do
    for batch_size in 4 8 16
    do
        for dropout in 0.2 0.3
        do
            for weight_decay in 0.1 0.01
            do
                python3 train.py --model="bert-base-uncased" --freeze --epochs=8 --dropout=$dropout --task="expla" --batch_size=$batch_size --lr=$lr --weight_decay=$weight_decay
                python3 train.py --model="bert-base-uncased" --freeze --epochs=8 --dropout=$dropout --task="expla" --use_graphs --batch_size=$batch_size --lr=$lr --weight_decay=$weight_decay
                python3 train.py --model="bert-base-uncased" --freeze --epochs=8 --dropout=$dropout --task="copa" --batch_size=$batch_size --lr=$lr --weight_decay=$weight_decay
                python3 train.py --model="bert-base-uncased" --freeze --epochs=8 --dropout=$dropout --task="copa" --use_graphs --batch_size=$batch_size --lr=$lr --weight_decay=$weight_decay
            done
        done
    done
done