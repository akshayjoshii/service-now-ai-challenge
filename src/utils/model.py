# Author: Akshay Joshi
# Email: akshayjoshi56@gmail.com
# Date: 08/09/2023

import torch
from torch import nn

from transformers import AutoModel

import logging
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)


class GTEModel(nn.Module):
    """ 
    Base class for General Text Embeddings (GTE) model
    Paper: Towards General Text Embeddings with Multi-stage Contrastive Learning
    Paper Link: https://arxiv.org/abs/2308.03281
    Pretrained Model Link: https://huggingface.co/thenlper/gte-base
    Additional Info: GTE model uses bert-base as the base model
    """
    def __init__(self, model_name_or_path:str):
        super(GTEModel, self).__init__()
        self.model_pth = model_name_or_path

        print("\nLoading the pretrained base model...")
        self.model = self._load_model()

    def _load_model(self):
        """ 
        Loads the pretrained model from the given path.

        :return: The pretrained model object
        """
        return AutoModel.from_pretrained(self.model_pth)

    def forward(
            self, 
            input_ids:torch.Tensor, 
            attention_mask:torch.Tensor, 
            token_type_ids:torch.Tensor
        ) -> torch.Tensor:
        """ Forward pass of the model. """
        
        output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
        
        # Get all the hidden states from the model
        hidden_states = output['hidden_states']

        # Average token embedding vectors from last 4 hidden layers to get 
        # a single averaged vector for each token. This is also called as 
        # Mean Pooling in the Sentence Transformers library.
        avg_hidden_states = torch.mean(
                                torch.stack(
                                    hidden_states[-4:]
                                ),
                                dim=0
                            )

        # Get the sentence embedding vector by taking the mean of the avg_hidden_states of tokens
        sentence_embedding = torch.mean(avg_hidden_states, dim=1)
        return sentence_embedding


class AkshayFormer(GTEModel):
    """ 
    Inherited class to modify any pretrained BERT/GTE-like models to add
    2 bottleneck adapter layers in each encoder block.
    The method is based on the paper:
    Parameter-Efficient Transfer Learning for NLP. Google Research. PMLR 20.
    Paper Link: https://arxiv.org/abs/1902.00751

    This class adds 4.23 million parameters to the GTE Base model - which has 110M
    parameters.
    """
    def __init__(self, model_name_or_path:str):
        super(AkshayFormer, self).__init__(model_name_or_path)
        
        # Add the bottleneck adapter in each encoder layer twice in 2 different
        # positions
        for layer in self.model.encoder.layer:
            
            # INSERTION: First position is after Attention & FFN
            layer.attention.output.add_module('bottleneck_adapter_1', 
                                            self.get_bottlneck_adapter())
            
            # Actual order after insertion: dense, layernorm, dropout, bottleneck_adapter
            # Desired order: dense -> bottleneck adapter -> layernorm -> dropout
            layer.attention.output._modules.move_to_end('bottleneck_adapter_1',
                                                        last=False)
            layer.attention.output._modules.move_to_end('dense', 
                                                        last=False)

            # INSERTION: Second position is after LayerNorm & FFN
            layer.output.add_module('bottleneck_adapter_2',
                                    self.get_bottlneck_adapter())
            
            # Desired order: dense -> bottleneck adapter -> layernorm -> dropout
            layer.output._modules.move_to_end('bottleneck_adapter_2',
                                                last=False)
            layer.output._modules.move_to_end('dense', 
                                              last=False)
        
    
    # Bottleneck Adapter contains 1 feedforward down projection (768->128), 
    # 1 non-linear activation & 1 feedforward up projection (128->768)
    def get_bottlneck_adapter(self):
        bottleneck_adapter = nn.Sequential(
                                nn.Linear(self.model.config.hidden_size, 
                                            self.model.config.hidden_size // 6),
                                nn.GELU(),
                                nn.Linear(self.model.config.hidden_size // 6, 
                                            self.model.config.hidden_size)
                            )
        return bottleneck_adapter