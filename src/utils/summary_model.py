# Author: Akshay Joshi
# Email: akshayjoshi56@gmail.com
# Date: 08/09/2023

import os
import torch
from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast, AutoTokenizer
)

from utils.preprocess_utils import (
                            create_text_chunks, 
                            tokenize_text_chunks
                        )

from typing import Union

# Set CUDA visible devices
os.environ["CUDA_VISIBLE_DEVICES"] = "4"


def load_device() -> str:
    """ 
    This function loads the device to be used for inference.

    :return: Device to be used for inference
    """

    device = "cuda:4" if torch.cuda.is_available() else "cpu"
    return device

def load_model(
        model_name_or_path:str="google/flan-t5-large"
    ) -> T5ForConditionalGeneration:
    """ 
    This function loads the model from the HuggingFace model hub.

    :param model_name_or_path: Name of the model to be loaded

    :return: T5ForConditionalGeneration model
    """

    model = T5ForConditionalGeneration.from_pretrained(
                model_name_or_path,
            )
    return model


def summarize_hierarchically(
        text_chunks:list,
        model:T5ForConditionalGeneration,
        tokenizer:Union[T5TokenizerFast, AutoTokenizer],
        summary_level:int=1,
        device:str="cpu",
        generation_hyperparams:dict={
            "max_length": 50,
            "num_beams": 3,
            "length_penalty": 1.0,
            "repetitions_penalty": 2.5,
            "early_stopping": True,
        }
    ) -> str:
    """
    This recursive function summarizes the text hierarchically. It first summarizes
    the text chunks individually. Then, it summarizes the summary of the chunks.

    :param text_chunks: List of text chunks
    :param model: T5ForConditionalGeneration model
    :param tokenizer: Tokenizer object
    :param summary_level: Level of summary. It can be 1, 2, or 3. [Default: 1]
            If 1, it summarizes the text chunks individually, then returns list of
                  summaries of dimension [num_chunks, max_length].
            If 2, it summarizes the text chunks individually, then recursively
                  summarizes the summaries of the chunks, then returns concatinated 
                  block of 2nd level summaries.
            If 3, it summarizes the text chunks individually, then recursively
                  summarizes the summaries of the chunks, then once again recursively
                  performs the same operation, then returns concatinated block of
                  3rd level summaries.
    :param device: Device to be used for inference. [Default: "cpu"]
    :param generation_hyperparams: Hyperparameters for the generation process.

    :return: 1 Concatinated block of individual summaries
    """

    # Get a generator object to yield tokenized text chunks
    tokenized_generator = tokenize_text_chunks(text_chunks, tokenizer)
        
    full_summary = ""
    for tokenized_item in tokenized_generator:
        output = model.generate(
                        tokenized_item.input_ids,
                        num_beams=generation_hyperparams['num_beams'],
                        max_length=generation_hyperparams['max_length'],  
                        repetition_penalty=generation_hyperparams['repetitions_penalty'], 
                        length_penalty=generation_hyperparams['length_penalty'], 
                        early_stopping=generation_hyperparams['early_stopping'],
                )

        decoded_output = tokenizer.decode(
                            output[0], 
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=True
                        )

        # Add a suffix full stop to the decoded output string, if it doesn't have one
        if decoded_output[-1] != ".":
            decoded_output += "."

        # If the decoded_output string has more than 10 tokens, then concat it 
        # to the full summary
        if len(decoded_output.split()) > 10:
            full_summary += decoded_output + " "
        

    # If summary level is 1, return the concatinated block 1st level summaries
    if summary_level == 1:
        return full_summary
    
    # If summary level is 2, summarize the summaries of the chunks
    elif summary_level == 2:
        # Create text chunks from the summaries of the chunks
        text_chunks = create_text_chunks(
                            long_text=full_summary,
                            tokenizer=tokenizer
                        )

        return summarize_hierarchically(
                    text_chunks,
                    model,
                    tokenizer,
                    summary_level=1,
                    device=device,
                    generation_hyperparams=generation_hyperparams
                )
    
    # If summary level is 3, summarize the summaries of summaries of the chunks
    elif summary_level == 3:
        # Create text chunks from the summaries of the chunks
        text_chunks = create_text_chunks(
                            long_text=full_summary,
                            tokenizer=tokenizer
                        )

        return summarize_hierarchically(
                    text_chunks,
                    model,
                    tokenizer,
                    summary_level=2,
                    device=device,
                    generation_hyperparams=generation_hyperparams
                )
    
    # If summary level is not 1, 2, or 3, raise an error
    else:
        raise ValueError("summary_level can only be 1, 2, or 3")



if __name__ == "__main__":
    print(load_device())
    print(load_model())








    

