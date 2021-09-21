# dogs-vs-cats

Author: Jeff Lin

---

## Summary

- [dogs-vs-cats](#dogs-vs-cats)
  - [Summary](#summary)
  - [Introduciotn](#introduciotn)
  - [Quick Run](#quick-run)
  - [Execution parameters](#execution-parameters)
  - [Experimental Results](#experimental-results)

---

## Introduciotn

This project is to practice the competition [Dogs vs. Cats Redux: Kernels Edition](#https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition) on Kaggle. The competition is a Dogs vs. Cats classification problem.

Submissions are scored on the log loss:

<img src="https://latex.codecogs.com/gif.latex?\text{LogLoss}=\frac{-1}{n}\sum\limits^n_{i=1}[y_i\log \hat{y}_i+(1-y_i)\log(1-\hat{y}_i)]" />

<img src="https://latex.codecogs.com/gif.latex?\text{LogLoss}=\frac{-1}{n}\sum\limits^n_{i=1}[y_i\log \hat{y}_i+(1-y_i)\log(1-\hat{y}_i)]" />

where

- <img src="https://latex.codecogs.com/gif.latex?n" /> is the number of images in the test set
- <img src="https://latex.codecogs.com/gif.latex?\hat{y}_i" /> is the predicted probability of the image being a dog
- <img src="https://latex.codecogs.com/gif.latex?y_i" /> is 1 if the image is a dog, 0 if cat
- <img src="https://latex.codecogs.com/gif.latex?\log" /> is the natural $base\ e$ logarithm

Submission File should have a header and be in the following format:

```text
id,label
1,0.5
2,0.5
3,0.5
...

```

---

## Quick Run

Run the following command to run default training and evaluating.

```python
python main.py
```

---

## Execution parameters

See detail by `python main.py -h` command.

- `-h, --help` : show this help message and exit
- `--cuda CUDA` : set CUDA device to traing and evaluating
- `--holdout-p HOLDOUT_P`
- `--num-workers NUM_WORKERS`
- `--batch-size BATCH_SIZE`
- `--epochs EPOCHS`
- `--model ['VGG19', 'VGG19_2', 'ResNet', 'MyCNN', 'Densenet', 'GoogleNet', 'inceptionv3']`
- `--iteration`
- `--train-all`
- `--optim OPTIM`
- `--lr LR`
- `--momentum MOMENTUM`
- `--scheduler`
- `--gamma GAMMA`
- `--threshold Float number >= 0 and <=1`
- `--output-path OUTPUT_PATH`
- `--train-path TRAIN_PATH`
- `--test-path TEST_PATH`
- `--submit-csv SUBMIT_CSV`
- `--kaggle Kaggle_Submission_Message` : Enter the submission message to upload Kaggle.

---

## Experimental Results

The submission score of the best experimental result is **0.06038**. Click [here](#[tmp](https://drive.google.com/file/d/14A-P7tUS1nfKbAs1Z3SvrFCovybxlBE7/view?usp=sharing)) to download the fine-trained model. The followings are the excution parameters of the best score.

|    Parameter    |  Value  |
| :-------------: | :-----: |
|  `--holdout-p`  |   0.8   |
| `--num-workers` |    8    |
| `--batch-size`  |    1    |
|   `--epochs`    |    1    |
|    `--model`    |  VGG19  |
|  `--iteration`  |  True   |
|  `--train-all`  |  True   |
|    `--optim`    |   SGD   |
|     `--lr`      |  1e-5   |
|  `--momentum`   |   0.9   |
|  `--scheduler`  |  True   |
|    `--gamma`    | 0.99985 |
|  `--threshold`  |  False  |
