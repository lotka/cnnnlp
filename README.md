# CNNNLP
Using CNNs for text classification [2] was the starting point for the code and [5] was used to pre train word embeddings

# Data
https://archive.ics.uci.edu/ml/datasets/Amazon+book+reviews

( More detail: http://ataspinar.com/2016/01/21/sentiment-analysis-with-bag-of-words/ )

# Usage Example
Assuming you extracted the amazon review datafiles to a folder called `csv`

Pretraining: Generate embeddings.pkl (example provided in this repo)
```
python train_embeddings.py
```

Training (creates a folder called runs):
```
python train.py --data_path='csv/'
```
By default use the first 60% of the data set. (Try `python train_embeddings.py --help` for more options.)


Evaluation:
```
python eval.py --data_path='csv/' --book_name='Suzanne-Collins-The-Hunger-Games.csv' --checkpoint_dir=runs/1488233333/checkpoints
```
If you don't supply the book_name it will by default use the last 40% of the full dataset

# Todo:
* Improve code so you can train without the word embeddings pickle
* See if larger models give accuracy improvements

# References
[1] Convolutional Neural Networks for Sentence Classification - https://arxiv.org/abs/1408.5882

[2] Implementing a CNN for Text Classification in TensorFlow - http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/

[3] Good literature overview: https://opendatascience.com/blog/understanding-convolutional-neural-networks-for-nlp/

[4] Word embeddings https://ireneli.eu/2017/01/17/tensorflow-07-word-embeddings-2-loading-pre-trained-vectors/

[5] word2vec implementation: https://www.tensorflow.org/code/tensorflow/examples/tutorials/word2vec/word2vec_basic.py
