"""
Generate qualitative side-by-side comparisons for Flickr8k.

This script reuses the greedy decoding recipe to load a LoRA adapter,
select a handful of sample indices, and visualize predictions vs.
ground-truth captions for rapid manual inspection.
"""

import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from transformers import (
    InstructBlipProcessor, 
    InstructBlipForConditionalGeneration, 
    BitsAndBytesConfig
)
from peft import PeftModel
from torch.utils.data import Dataset
import bitsandbytes as bnb
from bitsandbytes.nn import Linear4bit
import warnings

# 1. SETUP & MONKEY PATCH (Prevents Crashes)
warnings.filterwarnings("ignore")

# This patch fixes the 'NoneType' and 'KeyError' bugs in bitsandbytes
def safe_load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    # A. Handle Bias safely
    if self.bias is not None:
        bias_name = prefix + "bias"
        if bias_name in state_dict:
            bias_data = state_dict.pop(bias_name, None)
            if bias_data is not None:
                self.bias.data = bias_data.to(self.bias.data.device)
    
    # B. Handle Weights safely (Skip missing base weights)
    weight_name = prefix + "weight"
    if weight_name in state_dict:
        self.weight, state_dict = bnb.nn.Params4bit.from_state_dict(
            state_dict, prefix=weight_name + ".", requires_grad=False
        )
    
    # C. Cleanup
    unexpected_keys.extend(state_dict.keys())

# Apply the patch
Linear4bit._load_from_state_dict = safe_load_from_state_dict
print("Bitsandbytes compatibility patch applied.")

# 2. MODEL LOADING
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print("Loading Model")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

model_id = "Salesforce/instructblip-vicuna-7b"
processor = InstructBlipProcessor.from_pretrained(model_id, use_fast=False)

base_model = InstructBlipForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=quant_config,
    device_map={"": torch.cuda.current_device()}
)

# Load fine-tuned adapter (update to match your adapter artifact)
adapter_path = "/notebooks/final_adapter"
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()
print("Fine-tuned Model loaded successfully!")

# 3. DATASET CLASS
class Flickr8kEvalDataset(Dataset):
    def __init__(self, image_dir, caption_file, split="test"):
        self.image_dir = image_dir
        
        # Load CSV and normalize headers
        df = pd.read_csv(caption_file)
        df.columns = map(str.lower, df.columns)
        
        # Group by Image (1 image : List of 5 captions)
        self.grouped = df.groupby('image')['caption'].apply(list).reset_index()
        
        # Standard Split Logic
        unique_images = self.grouped['image'].tolist()
        if split == "train":
            target_imgs = unique_images[:6000]
        elif split == "val":
            target_imgs = unique_images[6000:7000]
        elif split == "test":
            target_imgs = unique_images[7000:]
            
        self.data = self.grouped[self.grouped['image'].isin(target_imgs)].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_name = row['image']
        references = row['caption'] # List of 5 strings
        image_path = os.path.join(self.image_dir, image_name)
        return image_name, image_path, references

# Initialize Dataset
# NOTE: Point these paths at your Flickr8k assets before running.
IMAGE_DIR = "/notebooks/flickr8k/Images"
CAPTION_FILE = "/notebooks/flickr8k/captions.txt"
eval_dataset = Flickr8kEvalDataset(IMAGE_DIR, CAPTION_FILE, split="test")
print(f"Dataset loaded: {len(eval_dataset)} test images.")

# 4. QUALITATIVE ANALYSIS GENERATOR
def show_qualitative_results(indices):
    print(f"\n{'='*20} QUALITATIVE ANALYSIS REPORT {'='*20}")
    
    # Set up the figure for plotting captions/refs side by side
    fig, axes = plt.subplots(1, len(indices), figsize=(15, 6))
    if len(indices) == 1: axes = [axes] # Handle single image case
    
    for i, idx in enumerate(indices):
        image_name, image_path, refs = eval_dataset[idx]
        image = Image.open(image_path).convert("RGB")
        
        # Generate Caption using your IMPROVED strategy (Greedy + Truncation)
        prompt = "Describe this image."
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=False,
                num_beams=1,          # Greedy Search (Crucial for concise captions)
                max_length=30,        # Hard limit
                min_length=5,
                repetition_penalty=2.0,
                length_penalty=1.0,
            )
        
        caption = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        
        # The "Period Trick" - Truncate after first sentence
        if "." in caption:
            caption = caption.split(".")[0] + "."
            
        # Print details to console
        print(f"\n🔹 Image Index: {idx}")
        print(f"   Ground Truth: {refs[0]}")
        print(f"   Model Pred:   {caption}")
        
        # Plot image
        axes[i].imshow(image)
        axes[i].axis("off")
        # Wrap text for title so it doesn't run off
        title_text = f"ID: {idx}\nPred: {caption}"
        axes[i].set_title(title_text, fontsize=10, wrap=True)

    plt.tight_layout()
    plt.show()

# 5. RUN ANALYSIS
# Indices requested: 1090 (Climber), 50 (Little Girl), 200 (Surfer)
indices_to_show = [1090, 50, 200] 
show_qualitative_results(indices_to_show)
