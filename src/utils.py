import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

def download_and_load_model(model_name="t5-small", device=None):
    """
    Downloads and loads a T5 model and tokenizer.
    Ensures proper error handling and caching.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        print(f"Downloading and loading {model_name}...")
        tokenizer = T5Tokenizer.from_pretrained(model_name, use_fast=True, cache_dir=None)
        model = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir=None)
        model.to(device)
        model.eval()
        print("Model and tokenizer loaded successfully.")
        return model, tokenizer, device
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying to redownload...")
        tokenizer = T5Tokenizer.from_pretrained(model_name, use_fast=True, force_download=True)
        model = T5ForConditionalGeneration.from_pretrained(model_name, force_download=True)
        model.to(device)
        model.eval()
        return model, tokenizer, device

