from src.utils.dataset_utils import (
    get_quora_dataset, 
    create_k_fold_datasets,
    load_df_from_pickle
)

from src.utils.preprocess_utils import normalize_text
from src.plot import plot_zipf_distribution


if __name__ == "__main__":
    quora = get_quora_dataset()

    # Split into 3 folds
    # create_k_fold_datasets(quora, num_folds=5)

    # Load df from pickle file
    df1 = load_df_from_pickle("data/cross_folds/train_1_folds.pkl")
    df2 = load_df_from_pickle("data/cross_folds/test_1_folds.pkl")

    # Plot Zipfs curve
    plot_zipf_distribution(
        dataframes=(df1, df2),
        title="Zipf's Curve for Custom Quora Dataset: Before Normalization", 
        save_path="plots/zipfs_curve_unnormalized.png"
    )

    df1 = normalize_text(df1)
    df2 = normalize_text(df2)
    plot_zipf_distribution(
        dataframes=(df1, df2),
        title="Zipf's Curve for Custom Quora Dataset: After Normalization", 
        save_path="plots/zipfs_curve_normalized.png"
    )


    