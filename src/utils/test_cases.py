# Author: Akshay Joshi
# Email: akshayjoshi56@gmail.com
# Date: 08/09/2023

import pandas as pd

# Assertion to check if the dataframe has 2 classes
def check_quora_class_balance(df:pd.DataFrame) -> None:
    """ 
    This function checks the class balance of the dataframe created for Quora Question Pair dataset.

    :param df: Pandas dataframe

    :return: None
    """

    assert len(df["is_duplicate"].unique()) == 2, "Dataframe does not have 2 classes"

    # Get class counts
    class_counts = df["is_duplicate"].value_counts()

    # Check if class counts are equal
    assert class_counts.iloc[0] == class_counts.iloc[1], "Dataframe is not balanced"