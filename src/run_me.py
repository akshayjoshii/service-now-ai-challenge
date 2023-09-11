from utils.dataset_utils import (
    get_quora_dataset, 
    create_k_fold_datasets,
    load_df_from_pickle
)

from utils.preprocess_utils import normalize_text
from utils.plot import plot_zipf_distribution

from utils.dataloader import get_dataset_generators
from utils.model import AkshayFormer
from utils.trainer import AdapterTransfomerTrainer


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

    # Get dataset generators
    train_dset, val_dset, test_dset = get_dataset_generators(
                                    train_df=df1,
                                    test_df=df2,
                                    model_name_or_path="thenlper/gte-base",
                                    max_seq_length=64,
                                    seed=55
                                )

    # Get the AkshayFormer model
    model = AkshayFormer(model_name_or_path="thenlper/gte-base")
    trainer = AdapterTransfomerTrainer(
                    model=model,
                    epochs=50,
                    learning_rate=0.05,
                    train_full_model=True,
                    model_save_name='AkshayFormer_AO_CV1',
            )
    
    # Get the dataloaders
    train_loader, test_loader, val_loader = trainer.get_data_loaders(
                                                train_dset,
                                                test_dset,
                                                val_dset,
                                                batch_size=256
                                            )
    
    # Train the model
    trainer.train(
            train_loader, 
            val_loader,
            test_loader
        )
    





    