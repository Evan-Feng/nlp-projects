# Usage

### 1. 数据

将train.txt，val.txt，test.txt三个文件放置在data/目录下

### 2. 生成词汇表

```
python gen_vocab.py
```

### 3. 训练模型

打印模型可选参数：

```
python train.py -h

usage: train.py [-h] [--train TRAIN] [--dev DEV] [--test TEST] [--vocab VOCAB]
                [--pretrained PRETRAINED] [--export EXPORT] [--seed SEED]
                [--cpu] [--emb_dim EMB_DIM] [--pos_dim POS_DIM]
                [--hidden_dim HIDDEN_DIM] [--num_layers NUM_LAYERS]
                [--dropout DROPOUT] [-bi [BIDIRECTIONAL]]
                [--regularization REGULARIZATION]
                [--attention {dot,general,concat}]
                [--label_smoothing LABEL_SMOOTHING] [--rnn_cell {GRU,LSTM}]
                [--balance] [--num_epochs NUM_EPOCHS] [-bs BATCH_SIZE]
                [-lr LEARNING_RATE] [--lr_decay LR_DECAY] [--lr_min LR_MIN]
                [-opt {SGD,Adam}] [--clip CLIP]
                [--teacher_forcing [TEACHER_FORCING]]
                [--val_metric {bleu-1,bleu-2,bleu-3,bleu-4} [{bleu-1,bleu-2,bleu-3,bleu-4} ...]]
                [--val_step VAL_STEP]
```

训练一个古诗生成模型，记录验证集上的BLEU-1和BLEU-2指标，并将模型储存在export/目录下：

```
python train.py --export export/ --val_metric bleu-1 bleu-2
```

注：默认采用GPU训练，如需使用CPU训练可指定命令行参数：

```
python train.py --cpu --export export/
```

### 4. 测试模型

因为测试集不完整，所以并没有实现评测脚本。模型训练完毕后会将测试集上生成的古诗储存到export目录下，可进行人工测评。