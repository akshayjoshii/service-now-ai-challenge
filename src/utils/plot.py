# Author: Akshay Joshi
# Email: akshayjoshi56@gmail.com
# Date: 08/09/2023

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Tuple

def get_word_counts(df:pd.DataFrame) -> dict:
    """
    This function takes a dataframe and returns a dictionary of word counts

    :param df: A dataframe with question1 and question2 columns
    :return: A dictionary of word counts
    """
    word_counts = {}

    cols = ['question1', 'question2']
    for col in cols:
        for sentence in df[col]:
            for word in sentence.split():
                if word not in word_counts:
                    word_counts[word] = 1
                else:
                    word_counts[word] += 1
    return word_counts

def plot_zipf_distribution(
        dataframes:Tuple[pd.DataFrame, pd.DataFrame],
        title:str='Zipf\'s Curve for Custom Quora Dataset',
        xlabel:str='Log Word Rank',
        ylabel:str='Log Word Frequency',
        save_path:str="/plots"
    ) -> None:
    """
    Plots the Zipfs curve for the given dataset

    :param dataframes: A tuple of train & test dataframes
    :param xlabel: Label for the x-axis
    :param ylabel: Label for the y-axis 
    :param save_path: Path to save the plot
    
    """
    plt.figure()

    # Get the word counts for the train and test dataframes
    train_word_counts = get_word_counts(dataframes[0])
    test_word_counts = get_word_counts(dataframes[1])

    
    # Sort the words by their frequency for both train & test
    sorted_word_counts_train = sorted(train_word_counts.items(), 
                                key=lambda x: x[1], reverse=True)
    
    sorted_word_counts_test = sorted(test_word_counts.items(),
                                key=lambda x: x[1], reverse=True)
    
    
    # Get the rank of each word
    word_ranks_train = [i for i in range(1, 
                len(sorted_word_counts_train) + 1)]
    

    word_ranks_test = [i for i in range(1,
                len(sorted_word_counts_test) + 1)]
    
    # Get the frequency of each word
    word_freqs_train = [x[1] for x in sorted_word_counts_train]
    word_freqs_test = [x[1] for x in sorted_word_counts_test]

    # Convert the word ranks to log scale
    log_word_ranks_train = np.log(word_ranks_train)
    log_word_ranks_test = np.log(word_ranks_test)

    # Convert the word frequencies to log scale
    log_word_freqs_train = np.log(word_freqs_train)
    log_word_freqs_test = np.log(word_freqs_test)

    # Draw grid lines
    plt.grid(True)

    # Plot both the Zipf's curves in a single plot & add 'o' markers
    plt.plot(
        log_word_ranks_train, 
        log_word_freqs_train, 
        label='Train Data', 
        color='orange',
    )
    plt.plot(
        log_word_ranks_test, 
        log_word_freqs_test, 
        label='Test Data', 
        color='blue',
    )

    plt.legend()

    plt.title(
            title,
            fontsize=10
        )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if save_path is not None:
        plt.savefig(save_path)

    print(f'Zipfs Distribution Plot is saved in: {save_path}')