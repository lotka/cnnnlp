# -*- coding: utf-8 -*-
import numpy as np
import re
import itertools
from collections import Counter
import pandas as pd
import os
import pandas as pd
import re
import os

def clean_str_simple(string):
    for c in '!"£$%^&*()_+{}:@~<>?|`-=[];\'\#\,\.\/\\':
        string = string.replace(c,'')
    return string.lower()

def load_book(book_path,max_sentence_length,encoding='binary'):
    """
        Load a book with pandas, applying various bits of preprocessing
    """
    df = pd.read_csv(book_path,delimiter='\t',header=None)

    # name the columns
    df.columns =  ['score','url','title','text']
    # drop the url
    df = df.drop('url',axis=1)
    # remove the html tags
    df.text = df.text.str.replace('<(.*?)>',' ')
    df.title = df.title.str.replace('<(.*?)>',' ')
    df.text = df.text.str.replace('/\s\s+/g',' ')
    df.title = df.title.str.replace('/\s\s+/g',' ')

    # clean up text
    df['title'] = df.title.apply(clean_str_simple).apply(lambda x : ' '.join(x.strip(' ').split(' ')[:max_sentence_length]))
    df['text'] = df.text.apply(clean_str_simple).apply(lambda x : ' '.join(x.strip(' ').split(' ')[:max_sentence_length]))
    df['onehot'] = encode_onehot(df.score,method=encoding)
    df = df[df.onehot.notnull()]

    return df

def load_books(path,books,max_sentence_length,balance_classes=False,encoding='full'):
    """
        Load a list of books
    """
    dfs = []
    for book in books:
        dfs.append(load_book(os.path.join(path,book),
                             max_sentence_length=max_sentence_length,
                             encoding=encoding))
    return pd.concat(dfs)


def encode_onehot(series,method='all'):
    """ encode a series into onehot format

        Args:
            series (pd.Series) : a pandas series with hopefully only a few distinct values

        Returns:
            pd.Series: series of lists
    """
    if method == 'full':
        # discover the categories
        categories = pd.np.sort(series.unique())
        # create onehot encoding
        return series.apply(lambda x : (categories == x).astype(int).tolist())
    if method == 'binary':
        def f(x):
            if x <= 2:
                return np.array([1,0])
            elif x >= 4:
                return np.array([0,1])
            else:
                return pd.np.nan

        return series.apply(f)

def load_data(max_sentence_length,
              path='csv',
              balance_classes=True,
              shuffle=True,
              encoding='binary',book=None):
    """
        Load all books and applying class balancing and shuffling
    """
    if book is None:
        books = ['EL-James-Fifty-Shades-of-Grey.csv',
                 'Andy-Weir-The-Martian.csv',
                 'Donna-Tartt-The-Goldfinch.csv',
                 'Fillian_Flynn-Gone_Girl.csv',
                 'John-Green-The-Fault-in-our-Stars.csv',
                 'Laura-Hillenbrand-Unbroken.csv',
                 'Paula_Hawkins-The-Girl-On-The-Train.csv',
                 'Suzanne-Collins-The-Hunger-Games.csv']
    else:
        books=[book]

    df = load_books(path,books,encoding=encoding,max_sentence_length=max_sentence_length,balance_classes=balance_classes)

    max_class_count = df.score.value_counts().loc[1]

    df_list = []
    for score,group in df.groupby(by='score'):
        if balance_classes:
            df_list.append(group.iloc[:max_class_count])
        else:
            df_list.append(group)

    df = pd.concat(df_list).reset_index(drop=True)

    if shuffle:
        np.random.seed(10)
        df = df.reindex(np.random.permutation(df.index)).reset_index(drop=True)

    return df

def dataframe_to_xy(df):
    """
        Convert pandas dataframe to tensorflow ready data strucutres
    """
    return df.text.values,np.vstack(df.onehot.values)

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
