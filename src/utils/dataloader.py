# Author: Akshay Joshi
# Email: akshayjoshi56@gmail.com
# Date: 08/09/2023

import torch
import pandas as pd
from transformers import AutoTokenizer
from torch.utils import data as torch_data

from typing import Union
from src.utils.test_cases import (
    check_if_dataframe, 
    check_if_exploded_df_cols_correct
)


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
                model_name_or_path:str,
                max_seq_length:int=128
            ):
        check_if_dataframe(dataset)
        check_if_exploded_df_cols_correct(dataset)
        
        # Just load dataframe with 3 columns
        self.dataset = dataset[["question1", "questions2", "is_duplicate"]]

        # Load the pretrained model's tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        return self.dataset.shape[0]


    def __getitem__(self, index:int) -> Union[
                                            dict(torch.Tensor), 
                                            dict(torch.Tensor), 
                                            torch.Tensor(int)
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
        # Each key is a list with 2 sublists: one for each ques in the pair
        return encoded_pair, label



        