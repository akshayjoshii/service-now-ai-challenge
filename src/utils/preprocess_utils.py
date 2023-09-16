# Author: Akshay Joshi
# Email: akshayjoshi56@gmail.com
# Date: 08/09/2023

import string
import swifter
import pandas as pd

from tqdm import tqdm
from transformers import T5TokenizerFast, AutoTokenizer

import nltk
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')

from typing import Union

""" 
Use 'swifter' for faster processing in all the functions below. 
"""

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


def load_tokenizer(
        tokenizer_name_or_path:str="google/flan-t5-large"
        ) -> Union[
                T5TokenizerFast,
                AutoTokenizer]:
    """
    This method loads the tokenizer specified by the user. By default, it uses the
    tokenizer for T5 model.

    :param tokenizer_name_or_path: Name or path of the tokenizer to be used.

    :return: Tokenizer object
    """

    try:
        # If the user picks the default T5 tokenizer
        tokenizer = T5TokenizerFast.from_pretrained(tokenizer_name_or_path)

    except:
        # When any other tokenizer is used
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    return tokenizer


def create_text_chunks(
        long_text:str,
        tokenizer:Union[
                        T5TokenizerFast,
                        AutoTokenizer
                    ]
    ) -> list[str]:
    """ 
    This method acepts really long input text & creates non-overlapping 
    chunks of text of length max_model_length. It also makes sure any sentence 
    is not split in between when chunk length is reached. In such cases, the
    sentence is added to the next chunk. 

    :param long_text: Input text
    :param tokenizer: Tokenizer object

    :return: List of text chunks
    """

    # Get the max length of the model
    max_model_length = tokenizer.model_max_length
    # print(f"Max seq. length of the model: {max_model_length}")

    sentences = nltk.tokenize.sent_tokenize(long_text)

    # initialize
    length = 0
    count = -1

    # Temp string to store the chunk
    chunk = "summarize: "

    # List to store all the chunks
    chunks = []

    # Wrap using tqdm for progress bar
    for sentence in tqdm(sentences, desc="Creating text chunks"):
        count += 1

        # Add the no. of sentence tokens to the length counter
        combined_length = len(tokenizer.tokenize(sentence)) + length

        # if it doesn't exceed the max length
        if combined_length  <= tokenizer.max_len_single_sentence: 
            chunk += sentence + " " # add the sentence to the chunk
            length = combined_length # update the length counter

            # if it is the last sentence
            if count == len(sentences) - 1:
                chunks.append(chunk) # save the chunk
            
        else: 
            chunks.append(chunk) # save the chunk
            
            # Reset the counters for a new chunk 
            length = 0 
            chunk = "summarize: "

            # Take care of the overflow sentence (i.e., sentence whose partial portion
            # can only be saved in the previous chunk)
            chunk += sentence + " "
            length = len(tokenizer.tokenize(sentence))

    return chunks


def tokenize_text_chunks(
        text_chunks:list[str],
        tokenizer:Union[
                        T5TokenizerFast,
                        AutoTokenizer
                    ]
    ) -> dict:
    """ 
    This method acepts a list of text chunks & tokenizes them using the tokenizer 
    specified by the user in batches. 

    :param text_chunks: List of text chunks
    :param tokenizer_obj: Tokenizer object

    :return: generator object which yields tokenized text chunks
    """

    for text_chunk in text_chunks:
        # Tokenize the text chunks
        tokenized_text_chunk = tokenizer(
                                    text_chunk,
                                    padding=False,
                                    truncation=True,
                                    return_tensors="pt",
                                    max_length=tokenizer.model_max_length,
                                )

        yield tokenized_text_chunk



if __name__ == "__main__":
    chunks = create_text_chunks("Akshay Joshi" * 200)
