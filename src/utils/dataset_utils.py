# Author: Akshay Joshi
# Email: akshayjoshi56@gmail.com
# Date: 08/09/2023

import pandas as pd
from datasets import load_dataset

from sklearn.model_selection import KFold

from src.utils.test_cases import check_quora_class_balance


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
        return quora.to_pandas()
    
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

        return quora








        
