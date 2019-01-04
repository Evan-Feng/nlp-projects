# Usage

### 1. 数据

将word.txt train.txt test.txt和test.answer.txt四个文件放置在data/目录下

### 2. 生成词汇表

```
python gen_vocab.py
```

### 3. 训练模型

打印模型可选参数：

```
usage: train.py [-h] [--train TRAIN] [--test TEST] [--vocab VOCAB]
                [--pretrained PRETRAINED] [--export EXPORT] [--seed SEED]
                [--cpu] [--emb_dim EMB_DIM] [--hidden_dim HIDDEN_DIM]
                [--num_layers NUM_LAYERS] [--dropout DROPOUT]
                [--max_length MAX_LENGTH] [-bi BIDIRECTIONAL]
                [--window_size WINDOW_SIZE] [--num_attention NUM_ATTENTION]
                [--regularization REGULARIZATION] [--num_epochs NUM_EPOCHS]
                [-bs BATCH_SIZE] [-lr LEARNING_RATE] [--lr_decay LR_DECAY]
                [--lr_min LR_MIN] [-opt {SGD,Adam}] [--clip CLIP]
                [--val_disable] [--val_fraction VAL_FRACTION]
                [--val_step VAL_STEP]
```

训练一个三层LSTM分词模型，并将模型储存在export/目录下：

```
python train.py --num_layers 3 --export export/
```

注：默认采用GPU训练，如需使用CPU训练可指定命令行参数：

```
python train.py --cpu --export export/
```

### 4. 测试模型

使用助教提供的perl脚本（需指定词汇表）：

```
perl score data/word.txt data/test.answer.txt export/prediction.txt
```

使用python脚本（无需指定词汇表）：

```
python eval.py export/prediction.txt
```

