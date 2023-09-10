# Author: Akshay Joshi
# Email: akshayjoshi56@gmail.com
# Date: 08/09/2023

import torch
import pandas as pd

from transformers import AutoTokenizer
from torch.utils import data as torch_data
from sklearn.model_selection import train_test_split

from utils.test_cases import (
    check_if_dataframe, 
    check_if_exploded_df_cols_correct
)

from typing import Union


class QQPDataset(torch_data.Dataset):
    """ 
    Custom torch dataset class to sample positive & negative samples from
    the Quora Question Pairs dataset.
    We are not performing any hardest negative mining here as the dataset
    already provides pos & neg pairs, unlike other STS datasets where only pos
    pairs for a reference sentence are provided.

    :param dataset: Pandas dataframe
    :param model_name_or_path: Pretrained model name or path
    :param max_seq_length: Maximum sequence length to be used for tokenization

    :return: None
    """
    def __init__(
                self, 
                dataset:pd.DataFrame,
                model_name_or_path:str="thenlper/gte-base",
                max_seq_length:int=128
            ):
        check_if_dataframe(dataset)
        check_if_exploded_df_cols_correct(dataset)
        
        # Just load dataframe with 3 columns
        self.dataset = dataset[["question1", "question2", "is_duplicate"]]

        # Load the pretrained model's tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        return self.dataset.shape[0]


    def __getitem__(self, index:int) -> Union[
                                            dict, 
                                            dict, 
                                            torch.Tensor
                                        ]:
        """
        Fetch tuple of 3 tensors for each index.
        
        In the original dataset, 1 is for duplicate question pairs and 0 is for
        non-duplicate question pairs. But to make this work for the 'CEL - Cosine
        Embedding Loss' in PyTorch, we need to convert 0 to -1.
        """

        # Tokenize the sampled question pair in batched mode
        encoded_pair = self.tokenizer(
                            [self.dataset.iloc[index]["question1"], 
                            self.dataset.iloc[index]["question2"]],
                            padding='max_length',
                            truncation=True,
                            max_length=self.max_seq_length,
                            return_tensors='pt'
                        )
        
        # Change label 0 to -1 & convert to tensor
        label = torch.tensor(
                    [1 if self.dataset.iloc[index]["is_duplicate"] == 1 else -1]
                )
        
        # encoded_pair is a dict with keys: input_ids, token_type_idsm attention_mask. 
        # Each key is a list with 2 sublists: one for each ques in the pair. 
        # Split this into 2 dicts: one for que1 & other for que2
        encoded_que1 = {k: v[0] for k, v in encoded_pair.items()}
        encoded_que2 = {k: v[1] for k, v in encoded_pair.items()}

        return encoded_que1, encoded_que2, label


def get_dataset_generators(
            train_df:pd.DataFrame,
            test_df:pd.DataFrame,
            model_name_or_path:str="thenlper/gte-base",
            max_seq_length:int=128,
            seed:int=7
    ) -> Union[
            torch_data.Dataset,
            torch_data.Dataset,
            torch_data.Dataset
        ]:
    """ 
    This utility function returns the train, validation and test dataset generators.
    
        Takes the train (40K) & test (10K) datasets as input.
        Train dataset is untouched.
        For Test & Val datasets, we split the test dataset into 2 
        parts: 5K & 5K (stratified, without replacement).

    :param train_df: Train dataframe
    :param test_df: Test dataframe

    :return: Tuple of train, validation and test dataset generators
    """

    # Create train dataset generator
    train_dataset = QQPDataset(
                        dataset=train_df,
                        model_name_or_path=model_name_or_path,
                        max_seq_length=max_seq_length
                    )

    # Split the test dataset into 2 parts: 5K & 5K (stratified, without replacement)
    # Use sklearn train_test_split with stratify
    test_dset, val_dset = train_test_split(
                            test_df, 
                            test_size=0.5, 
                            stratify=test_df["is_duplicate"],
                            random_state=seed
                        )
    
    # Check if both the datasets have equal samples
    assert test_dset.shape[0] == val_dset.shape[0], \
            "Test and Val datasets do not have equal no. of samples"

    # Create test & val dataset generators
    test_dataset = QQPDataset(
                        dataset=test_dset,
                        model_name_or_path=model_name_or_path,
                        max_seq_length=max_seq_length
                    )
    
    val_dataset = QQPDataset(
                        dataset=val_dset,
                        model_name_or_path=model_name_or_path,
                        max_seq_length=max_seq_length
                    )

    return train_dataset, val_dataset, test_dataset
    
    



        