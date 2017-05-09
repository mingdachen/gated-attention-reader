# GA-Reader
Tensorflow implementation of [Gated Attention Reader for Text Comprehension](https://arxiv.org/abs/1606.01549). The original code can be found from [here](https://github.com/bdhingra/ga-reader). For pytorch implementation, please check pytorch branch.

## Prerequisites
- Python 3.5
- TensorFlow 1.0+
- tqdm

## Preprocessed Data
You can get the preprocessed data files from [here](https://drive.google.com/drive/folders/0B7aCzQIaRTDUZS1EWlRKMmt3OXM?usp=sharing). 

You can also get the pretrained Glove vectors from the above link.

## Training

```
python main.py --data_dir ~/data/dailymail --embed_file ~/data/word2vec_glove.txt
```

You should see around 0.7 accuracy on training set after running 1000 iteration (total 27486 iterations) on dailymail with default setting.