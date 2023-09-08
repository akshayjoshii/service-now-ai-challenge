# Author: Akshay Joshi
# Email: akshayjoshi56@gmail.com
# Date: 08/09/2023

import pandas as pd
from datasets import load_dataset

from sklearn.model_selection import KFold

from src.utils.test_cases import check_quora_class_balance


def explode_quora_dataframe(
        quora:pd.DataFrame
    ) -> pd.DataFrame:
    """
    This function takes a quora dataframe and returns a new dataframe with new columns.
    Old DF columns: questions, is_duplicate
    New DF columns: id1, id2, question1, question2, is_duplicate
    """
    
    # Create a new DataFrame with the desired column names
    new_df = pd.DataFrame(columns=[
                        'id1', 
                        'id2', 
                        'question1', 
                        'question2', 
                        'is_duplicate'
                    ]
                )
    
    # Extract id, text columns from the original DataFrame
    # Split the id and text values into separate columns
    id_values = quora['questions'].apply(lambda x: x['id'])
    new_df['id1'] = id_values.apply(lambda x: x[0])
    new_df['id2'] = id_values.apply(lambda x: x[1])

    text_values = quora['questions'].apply(lambda x: x['text'])
    new_df['question1'] = text_values.apply(lambda x: x[0])
    new_df['question2'] = text_values.apply(lambda x: x[1])
    
    # Copy the 'is_duplicate' column
    new_df['is_duplicate'] = quora['is_duplicate']
    return new_df


def get_quora_dataset(
        num_samples:int=50000,
        balanced:bool=True,
        rand_seed:int=7
    ) -> pd.DataFrame:
    """ 
    This function loads the quora dataset from huggingface datasets and 
    returns a pandas dataframe with num_samples samples.

    :param num_samples: Number of samples to return
    :param balanced: If True, return equal number of samples with is_duplicate = True & False
    :param rand_seed: Random seed for reproducibility

    :return: Pandas dataframe with num_samples samples
    """

    # Load quora dataset from huggingface datasets
    quora = load_dataset(
                "quora", 
                split="train", 
                cache_dir="data/"
            ).shuffle(seed=rand_seed)
    
    # In both cases, sampling without replacement is done
    if not balanced:
        # Get num_samples samples without bothering about class balance
        quora = quora.select(range(num_samples))
        return explode_quora_dataframe(quora.to_pandas())
    
    else:
        # Convert to dataframe
        quora = quora.to_pandas()

        # Without replacement, sample 50% of num_samples with is_duplicate = True & 
        # rest 50% of num_samples with is_duplicate = False
        quora = quora.groupby("is_duplicate").apply(
                    lambda x: x.sample(
                        n=num_samples//2, 
                        replace=False, 
                        random_state=rand_seed
                    )
        )

        # Shuffle the dataframe
        quora = quora.sample(frac=1, random_state=rand_seed).reset_index(drop=True)

        # Run a few test cases to check if the dataframe is balanced
        check_quora_class_balance(quora)

        return explode_quora_dataframe(quora)


def get_k_fold_datasets(
        dataset:pd.DataFrame,
        num_folds:int=3,
        rand_seed:int=7):
    """
    This function splits the dataset into num_folds folds and returns a list of
    tuples (train_df, test_df) where each tuple represents a fold.
    """

    # Initialize KFold object with num_folds folds
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=rand_seed)

    # Initialize list of tuples
    k_fold_datasets = []

    # Iterate over the folds
    for train_index, test_index in kf.split(dataset):
        # Get train and test dataframes
        train_df = dataset.iloc[train_index]
        test_df = dataset.iloc[test_index]

        # Append to list
        k_fold_datasets.append((train_df, test_df))

    return k_fold_datasets








        
