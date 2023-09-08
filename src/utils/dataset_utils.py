# Author: Akshay Joshi
# Email: akshayjoshi56@gmail.com
# Date: 08/09/2023

import pandas as pd
from datasets import load_dataset

from sklearn.model_selection import KFold

from src.utils.test_cases import (
    check_quora_class_balance,
    check_if_dataframe,
    check_if_exploded_df_cols_correct,
    check_data_leakage
)


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
    check_if_exploded_df_cols_correct(new_df)
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


def create_k_fold_datasets(
        dataset:pd.DataFrame,
        num_folds:int=5,
        rand_seed:int=7) -> None:
    """
    This function splits the dataset into num_folds folds and saves each fold as a pickle file.

    :param dataset: Pandas dataframe to split into folds
    :param num_folds: Number of folds to split the dataset into
    :param rand_seed: Random seed for reproducibility

    :return: None

    """

    # Initialize KFold object with num_folds folds
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=rand_seed)

    # Iterate over the folds
    idx = 1
    for train_index, test_index in kf.split(dataset):
        # Get train and test dataframes
        train_df = dataset.iloc[train_index].reset_index(drop=True)
        test_df = dataset.iloc[test_index].reset_index(drop=True)
        
        # Test cases to check if the dataframes are valid & there is no data leakage
        check_if_dataframe(train_df)
        check_if_dataframe(test_df)
        check_data_leakage(train_df, test_df)

        # dump the train and test dataframes as pickle files
        train_df.to_pickle(f"data/cross_folds/train_{idx}_folds.pkl")
        test_df.to_pickle(f"data/cross_folds/test_{idx}_folds.pkl")
        idx += 1
    
    print(f"Created {num_folds} folds of the dataset and saved them as pickle files in data/cross_folds/")


def load_df_from_pickle(
        file_path:str
    ) -> pd.DataFrame:
    """
    This function loads a dataframe from a pickle file.

    :param file_path: Path of the pickle file

    :return: Pandas dataframe
    """

    df = pd.read_pickle(file_path)
    check_if_dataframe(df)
    return df

        











        
