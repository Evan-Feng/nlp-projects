# semparse

This repository contains semantic parsing model for single-relation questions and questions with CVT structures.

## Usage

### Data Preprocessing

```bash
git clone https://github.com/Evan-Feng/nlp-projects.git
cd nlp-projects/semparse
```

By default, files EMNLP.train, EMNLP.dev, EMNLP.test are already placed them under the folder "data/". To preprocess the data, run the following command:

```bash
python preprocess.py
```

This will split the data into two single-relation (sgl) questions and CVT-structured (cvt) questions and binarize the data. The resulting file structure will look like:

```
.
├── README.md
└── data
    ├── EMNLP.train
    ├── EMNLP.dev
    ├── EMNLP.test
    ├── sgl_train.pth    (binarized single-relation train set)
    ├── sgl_dev.pth      (binarized single-relation dev set)
    ├── sgl_test.pth     (binarized single-relation test set)
    ├── cvt_train.pth    (binarized cvt train set)
    ├── cvt_dev.pth      (binarized cvt dev set)
    └── cvt_test.pth     (binarized cvt test set)
```

### Parsing Questions with Single Relation

To train a semantic parsing model for single-relation questions and save it to a specified location:

```
python single_relation.py --export export/sgl0/
```

To get predictions on the test set using the trained model:

```
python single_relation.py --export export/sgl0/ --mode eval --test data/EMNLP.test --output data/sgl0.pred
```

### Parsing Questions with CVT structures

To train a semantic parsing model for single-relation questions and save it to a specified location:

```
python cvt.py --export export/cvt0/
```

To get predictions on the test set using the trained model:

```
python cvt.py --export export/cvt0/ --mode eval --test data/EMNLP.test --output data/cvt0.pred
```

### Combine Predictions from the Two Models

Since the questions of both types are contained in a single test file (EMNLP.test), it may be convenient to combined the predictions from the two models in order to directly compare the predictions with the ground truth. To do this, simply run:

```bash
python merge_pred.py --sgl data/sgl0.pred --cvt data/cvt0.pred --output data/predictions.txt
```

