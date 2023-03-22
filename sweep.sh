
for lr in 1e-5 8e-6 6e-6 4e-6 2e-6 1e-6
do
    for batch_size in 4 8 16 32 64
    do
        python3 train.py --seed=$seed --model="roberta-large" --epochs=10 --task="copa" --batch_size=$batch_size --lr=$lr
    done
done