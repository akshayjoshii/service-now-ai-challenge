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


# Assertion to check if the object is a pandas dataframe
def check_if_dataframe(df:pd.DataFrame) -> None:
    """ 
    This function checks if the object is a pandas dataframe.

    :param df: Object to check

    :return: None
    """

    assert isinstance(df, pd.DataFrame), "Object is not a pandas dataframe"

# Assertion to check if the exploded df has the desired columns
def check_if_exploded_df_cols_correct(df:pd.DataFrame) -> None:
    """ 
    This function checks if the exploded dataframe has the desired columns.

    :param df: Exploded dataframe

    :return: None
    """

    assert df.columns.tolist() == [
        'id1', 
        'id2', 
        'question1', 
        'question2', 
        'is_duplicate'
    ], "Exploded dataframe does not have the desired columns"

# Check if there is data leakage between train and test sets
def check_data_leakage(df1:pd.DataFrame, df2:pd.DataFrame) -> None:
    """ 
    This function checks if there is data leakage between train and test sets.

    :param df1: Train dataframe
    :param df2: Test dataframe

    :return: None
    """

    # Convert the id1, id2 columns to tuples
    df1['id'] = df1[['id1', 'id2']].apply(tuple, axis=1)
    df2['id'] = df2[['id1', 'id2']].apply(tuple, axis=1)

    # Convert the id column to a set
    df1_ids = set(df1['id'])
    df2_ids = set(df2['id'])

    # Drop the id column
    df1.drop('id', axis=1, inplace=True)
    df2.drop('id', axis=1, inplace=True)

    # Check if there is data leakage
    assert len(df1_ids.intersection(df2_ids)) == 0, "There is data leakage between train and test sets"