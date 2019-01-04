for dp in 0.5 0.2 0.7
do
    for lr in 0.003
    do
        for nl in 3 1 2 
        do
            bs=5000
            python -u train.py -bs $bs --emb_dim 128 --dropout $dp --num_layers $nl --val_fraction 0.05 --num_epochs 800 -lr 0.003 --export export/train-dp$dp-nl$nl-bs$bs/
            bs=500
            python -u train.py -bs $bs --emb_dim 128 --dropout $dp --num_layers $nl --val_fraction 0.05 --num_epochs 600 -lr 0.003 --export export/train-dp$dp-nl$nl-bs$bs/
            bs=50
            python -u train.py -bs $bs --emb_dim 128 --dropout $dp --num_layers $nl --val_fraction 0.05 --num_epochs 400 -lr 0.001 --export export/train-dp$dp-nl$nl-bs$bs/
        done
    done
done
