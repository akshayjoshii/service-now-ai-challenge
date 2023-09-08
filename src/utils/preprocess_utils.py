# Author: Akshay Joshi
# Email: akshayjoshi56@gmail.com
# Date: 08/09/2023

import string
import swifter
import pandas as pd

import nltk
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')

""" Use 'swifter' for faster processing in all the functions below. """

def encode_labels(df:pd.DataFrame) -> pd.DataFrame:
    """ 
    This function encodes the labels in the column is_duplicate.
    False (Not duplicate) -> 0
    True (Duplicate) -> 1

    :param df: Pandas dataframe

    :return: Pandas dataframe
    """

    df['is_duplicate_encoded'] = df['is_duplicate'].swifter.apply(
                                lambda x: 1 if x == True else 0)
    return df


def lower_case(df:pd.DataFrame) -> pd.DataFrame:
    """ 
    This function converts the text in the columns question1 and question2 to lower case.

    :param df: Pandas dataframe

    :return: Pandas dataframe
    """

    df['question1'] = df['question1'].swifter.apply(lambda x: x.lower())
    df['question2'] = df['question2'].swifter.apply(lambda x: x.lower())
    return df

def remove_punc_special_tokens(df:pd.DataFrame) -> pd.DataFrame:
    """ 
    This function removes the punctuation and special tokens from the text in the columns question1 and question2.

    :param df: Pandas dataframe

    :return: Pandas dataframe
    """

    df['question1'] = df['question1'].swifter.apply(
                        lambda x: x.translate(str.maketrans('', '', string.punctuation))
                    )
    df['question2'] = df['question2'].swifter.apply(
                        lambda x: x.translate(str.maketrans('', '', string.punctuation))
                    )
    return df

def remove_stopwords(df:pd.DataFrame) -> pd.DataFrame:
    """ 
    This function removes the stopwords from the text in the columns question1 and question2.

    :param df: Pandas dataframe

    :return: Pandas dataframe
    """

    df['question1'] = df['question1'].swifter.apply(
                        lambda x: ' '.join([word for word in x.split() if word not in (stopwords)])
                    )
    df['question2'] = df['question2'].swifter.apply(
                        lambda x: ' '.join([word for word in x.split() if word not in (stopwords)])
                    )
    return df

def normalize_text(df:pd.DataFrame) -> pd.DataFrame:
    """ 
    This function normalizes the text in the columns question1 and question2.

    :param df: Pandas dataframe

    :return: Pandas dataframe
    """
    df = encode_labels(df)
    df = lower_case(remove_stopwords(remove_punc_special_tokens(df)))

    # Drop rows with empty strings
    df = df[df['question1'] != '']
    df = df[df['question2'] != '']

    return df