# Author: Akshay Joshi
# Email: akshayjoshi56@gmail.com
# Date: 08/09/2023

import os

import torch
from tqdm import tqdm

from transformers import AutoModel
from torch.optim import lr_scheduler
from torch.utils import data as torch_data

from sklearn.metrics import classification_report

from typing import Union
# from accelerate import Accelerator

# Track training progress using wandb
import wandb
from dotenv import load_dotenv

# Load keys from .env file
load_dotenv(dotenv_path='.env')
WANDB_KEY = os.getenv('WANDB_KEY')

print('\nLogging training progress using wandb...'
        'Please make sure to login to wandb using your API key.')

try:
    wandb.login(key=WANDB_KEY)
except:
    print('Unable to login to wandb. Please check your API key.'
          'And put your key in the .env file in src directory.')


# Set visible CUDA devices
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Init HF Accelerate
# accelerator = Accelerator()


class AdapterTransfomerTrainer(object):
    def __init__(
            self, 
            model:AutoModel,
            epochs:int, 
            learning_rate:float,
            train_full_model:bool=False,
            model_save_name:str='AkshayFormer_Full_CV1',
            model_save_path:str=os.path.join(
                                    "..", 
                                    "data", 
                                    "finetuned_models"
                                )
        ) -> None:
        """
        Class to handle the training of the model for the STS task over QQP 
        dataset using metric learning loss.

        :param model: Pretrained model object
        :param epochs: Number of epochs
        :param learning_rate: Learning rate
        :param device: List of GPU IDs to use
        :param train_full_model: Whether to finetune the full model (incl. Adapter
                                    layers) or just the Adapter layers!

        :return: None
        """
        
        self.model = model
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.train_full_model = train_full_model

        # Model save path
        self.model_save_name = model_save_name
        self.model_save_path = model_save_path

        # TODO: Use HF Accelerate to train on multiple GPUs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Print on which device the model is being trained
        print(f'\nTraining the model on: {self.device}')

        # Wandb project details
        self.wandb_project = 'servicenow-ai-challenge'
        self.wandb_entity = 'akshay-joshi'

        # Init the wandb project
        wandb.init(
                project=self.wandb_project, 
                entity=self.wandb_entity,
                config={
                    'epochs': self.epochs,
                    'learning_rate': self.learning_rate,
                    'train_full_model': self.train_full_model,
                    'batch_size': 3000,
                    'cross_fold_id': 2
                    },
                dir="logs"
                )

        # If False, only update the gradients of the adapter layers
        if not self.train_full_model:
            print("\n", "-" * 10) 
            print("\nOnly 4.7M parameters will be finetuned instead of 115 M params. of the full model (110M in base model + 4.7M in adapter blocks)")
            print("\n", "-" * 10)

            for name, param in self.model.named_parameters():
                if 'adapter' not in name:
                    param.requires_grad = False

        else:
            print("\n-" * 10) 
            print("\nAll the 114M parameters of the model will be finetuned!")
            print("\n-" * 10)

        # Total number of trainable parameters in the model
        print('\nTotal number of trainable parameters in the model: ' 
              f'{self.count_parameters()}')

        # Init the AdamW optimizer function
        self.optimizer = torch.optim.AdamW(
                                    self.model.parameters(),
                                    lr=self.learning_rate,
                                    eps=1e-6,
                                    weight_decay=0.01,
                                    amsgrad=True
                                )

        # Scheduler with cosine annealing scheme
        self.scheduler = lr_scheduler.CosineAnnealingLR(
                                                    self.optimizer,
                                                    T_max=self.epochs,
                                                    eta_min=1e-8
                                                )

        # Init Cosine Embedding Loss from PyTorch
        self.margin = 0.5
        self.cosine_loss = torch.nn.CosineEmbeddingLoss(margin=self.margin)

        # Track the gradients of the model
        wandb.watch(self.model, log='all')

        # Move the model to the device
        self.model.to(self.device)


    def get_data_loaders(
            self,
            train_dataset:torch_data.Dataset,
            test_dataset:torch_data.Dataset,
            val_dataset:torch_data.Dataset,
            batch_size:int=32,
            num_workers:int=2,
            pin_memory:bool=True,
            shuffle:bool=True,
        ) -> Union[
                torch_data.DataLoader, 
                torch_data.DataLoader,
                torch_data.DataLoader]:

        """ 
        Create the dataloaders for train, val & test datasets.
        
        :param train_dataset: Train dataset
        :param test_dataset: Test dataset
        :param val_dataset: Validation dataset
        :param batch_size: Batch size
        :param num_workers: Number of CPU workers/threads
        :param pin_memory: Whether to pin the memory to GPU
        :param shuffle: Whether to shuffle the data

        :return: train, test and val dataloaders
        """

        # Create the dataloaders
        dataloaders = []
        for dataset in [train_dataset, test_dataset, val_dataset]:
            dataloaders.append(
                torch_data.DataLoader(
                    dataset=dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    shuffle=shuffle,
                )
            )
        
        return dataloaders[0], dataloaders[1], dataloaders[2]
        

    def count_parameters(self) -> int:
        """ Count the total number of parameters in the model. """
        return sum(p.numel() for p in self.model.parameters() 
                    if p.requires_grad)
                                            

    def train(
            self, 
            train_dataloader:torch_data.DataLoader, 
            val_dataloader:torch_data.DataLoader,
            test_dataloader:torch_data.DataLoader
        ) -> None:
        """ 
        Finetune the model using metric learning loss & evaluate it on the test set.
        Metrics used for evaluation: Accuracy & F1 score.

        :param train_dataloader: Train dataloader
        :param val_dataloader: Validation dataloader
        :param test_dataloader: Test dataloader

        :return: None
        """

        if train_dataloader is None or val_dataloader is None:
            raise ValueError('Either of train_dataloader and val_dataloader \
                            cannot be empty')

        print(f'\nFinetuning the model for {self.epochs} epochs...')

        with tqdm(total=self.epochs, desc='Epochs') as pbar:
            for epoch in range(self.epochs):
                # Training
                train_loss = self.train_epoch(train_dataloader)

                # Update the LR Scheduler
                self.scheduler.step()

                # Validation
                val_loss = self.validate(val_dataloader)

                # Print the loss values
                print(f'Epoch: {epoch+1}/{self.epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}')

                pbar.update(1)

        print("\nEvaluating the model on the test set, please wait...")
        classification_report = self.test(test_dataloader)

        # Finish the wandb run
        wandb.finish()

        # Save the model
        self.save_model(
                model_name=self.model_save_name,
                model_path=self.model_save_path
            )


    def train_epoch(
                self, 
                train_dataloader:torch_data.DataLoader
            ) -> float:
        """ 
        Train the model for one epoch & return the loss. 

        :param train_dataloader: Train dataloader

        :return: Average loss for the epoch
        """
        self.model.train()

        total_loss = 0
        for batch in train_dataloader:
            
            # Unpack the batch
            encoded_que1, encoded_que2, label = batch

            # Move the data to the device
            label = label.to(self.device, 
                            dtype=torch.long).squeeze(1)
            q1_ids = encoded_que1['input_ids'].to(self.device,
                                            dtype=torch.long).squeeze(1)
            q1_mask = encoded_que1['attention_mask'].to(self.device,
                                            dtype=torch.long).squeeze(1)
            
            q1_type_ids = encoded_que1['token_type_ids'].to(self.device,
                                            dtype=torch.long).squeeze(1)
            
            q2_ids = encoded_que2['input_ids'].to(self.device,
                                            dtype=torch.long).squeeze(1)
            
            q2_mask = encoded_que2['attention_mask'].to(self.device,
                                            dtype=torch.long).squeeze(1)

            q2_type_ids = encoded_que2['token_type_ids'].to(self.device,
                                            dtype=torch.long).squeeze(1)
            
            # Get the embeddings from the model
            q1_embeddings = self.model(
                                q1_ids,
                                q1_mask,
                                q1_type_ids
                            )
            
            q2_embeddings = self.model(
                                q2_ids,
                                q2_mask,
                                q2_type_ids
                            )
            
            # q1_emb.shape = [64, 768]
            # q2_emb.shape = [64, 768]
            # label.shape = [64, 1]


            # Compute the cosine loss
            loss = self.cosine_loss(q1_embeddings, q2_embeddings, label)

            loss.requires_grad = True
            loss.backward()

            self.optimizer.zero_grad()
            self.optimizer.step()

            # Detach the loss from the graph & send to cpu
            loss = loss.detach().cpu()

            # Add the loss to the total loss
            total_loss += loss.item()

            # Track all the losses using wandb
            wandb.log({'total_train_loss': loss.item(),
                       'lr': self.scheduler.get_last_lr()[0]})

        return total_loss / len(train_dataloader)


    def validate(self, val_dataloader:torch_data.DataLoader) -> float:
        """ Validate the model. """
        self.model.eval()

        total_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                
                # Unpack the batch
                encoded_que1, encoded_que2, label = batch

                # Move the data to the device
                label = label.to(self.device, dtype=torch.long).squeeze(1)
                q1_ids = encoded_que1['input_ids'].to(self.device,
                                                dtype=torch.long).squeeze(1)
                q1_mask = encoded_que1['attention_mask'].to(self.device,
                                                dtype=torch.long).squeeze(1)
                
                q1_type_ids = encoded_que1['token_type_ids'].to(self.device,
                                                dtype=torch.long).squeeze(1)
                
                q2_ids = encoded_que2['input_ids'].to(self.device,
                                                dtype=torch.long).squeeze(1)
                
                q2_mask = encoded_que2['attention_mask'].to(self.device,
                                                dtype=torch.long).squeeze(1)

                q2_type_ids = encoded_que2['token_type_ids'].to(self.device,
                                                dtype=torch.long).squeeze(1)
                
                # Get the embeddings from the model
                q1_embeddings = self.model(
                                    q1_ids,
                                    q1_mask,
                                    q1_type_ids
                                )
                
                q2_embeddings = self.model(
                                    q2_ids,
                                    q2_mask,
                                    q2_type_ids
                                )

                # Compute the cosine loss
                loss = self.cosine_loss(q1_embeddings, q2_embeddings, label)

                # Detach the loss from the graph & send to cpu
                loss = loss.detach().cpu()
                total_loss += loss.item()

                # Track all the losses using wandb
                wandb.log({'total_val_loss': loss.item(),
                            'lr': self.scheduler.get_last_lr()[0]})
                
        return total_loss / len(val_dataloader)

    
    def test(
            self,
            test_dataloader:torch_data.DataLoader
        ) -> dict:
        """
        Use the final finetuned model to evaluate it's performance on the test set.
        For eval on QQP dataset we use Accuracy & F1 from Sklearn's classification_report.

        :param test_dataloader: Test dataloader 

        :return: Classification report dict
        """
        self.model.eval()

        with torch.no_grad():
            for batch in test_dataloader:
                
                # Unpack the batch
                encoded_que1, encoded_que2, label = batch

                # Change the range of labels back from -1, 1 to 0, 1 (had did this
                # to use the CosineEmbeddingLoss)
                label = [1 if l == 1 else 0 for l in label]

                # Move the data to the device
                q1_ids = encoded_que1['input_ids'].to(self.device,
                                                dtype=torch.long).squeeze(1)
                q1_mask = encoded_que1['attention_mask'].to(self.device,
                                                dtype=torch.long).squeeze(1)
                
                q1_type_ids = encoded_que1['token_type_ids'].to(self.device,
                                                dtype=torch.long).squeeze(1)
                
                q2_ids = encoded_que2['input_ids'].to(self.device,
                                                dtype=torch.long).squeeze(1)
                
                q2_mask = encoded_que2['attention_mask'].to(self.device,
                                                dtype=torch.long).squeeze(1)

                q2_type_ids = encoded_que2['token_type_ids'].to(self.device,
                                                dtype=torch.long).squeeze(1)
                
                # Get the embeddings from the model
                q1_embeddings = self.model(
                                    q1_ids,
                                    q1_mask,
                                    q1_type_ids
                                )
                
                q2_embeddings = self.model(
                                    q2_ids,
                                    q2_mask,
                                    q2_type_ids
                                )

                # Compute cosine similarity between the embeddings using torch
                cosine_sim = torch.cosine_similarity(
                                q1_embeddings,
                                q2_embeddings,
                                dim=1
                            )

                # Detach the cosine_sim from the graph & send to cpu
                cosine_sim = cosine_sim.detach().cpu().numpy()

                # Convert the cosine_sim to 0s & 1s
                cosine_sim = [1 if sim >= self.margin else 0 for sim in cosine_sim]

                # Get the classification report
                classification_report_dict = classification_report(
                                                label,
                                                cosine_sim,
                                                output_dict=True
                                            )

                # Log the classification report using wandb
                wandb.log({'classification_report': classification_report_dict})
                

    def save_model(
            self, 
            model_name:str='AkshayFormer',
            model_path:str=os.path.join(
                                    "data", 
                                    "finetuned_models"
                                )
        ) -> None:
        """ Save the finetuned model & its weights. """
        
        model_path = os.path.join(model_path, model_name)

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # Save the model weight & state dict to the model path
        torch.save(
            self.model.state_dict(), 
            os.path.join(model_path, 'model_weights.pth'))
        
        print(f"Model is saved in: {model_path}\n")