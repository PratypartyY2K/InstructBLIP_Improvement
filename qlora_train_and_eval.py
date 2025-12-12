"""
InstructBLIP Fine-Tuning & Evaluation on Flickr8k

Description:
    This script performs end-to-end QLoRA fine-tuning of the InstructBLIP (Vicuna-7B) model
    on the Flickr8k dataset. It includes:
    1. Automatic download of the Karpathy Split JSON.
    2. A custom Dataset class handling Prompt+Caption concatenation.
    3. 4-bit QLoRA training (Low-Rank Adaptation).
    4. Evaluation on the test split using standard metrics (BLEU-4, CIDEr).

Usage:
    pip install -r requirements.txt
    python main_flickr8k.py

Author: Pratyush Kumar
Date: December 2025
"""

import os
import json
import gc
import torch
import requests
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import (
    InstructBlipProcessor, 
    InstructBlipForConditionalGeneration,
    TrainingArguments, 
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

# Configuration
class Config:
    # Paths
    DATASET_DIR = '/content/drive/MyDrive/flickr8k_dataset/Images'
    KARPATHY_JSON = 'dataset_flickr8k.json'
    OUTPUT_DIR = '/content/drive/MyDrive/instructblip-flickr8k-lora'
    
    # Model Settings
    MODEL_ID = "Salesforce/instructblip-vicuna-7b"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Training Hyperparameters
    EPOCHS = 3
    BATCH_SIZE = 4
    GRAD_ACCUMULATION = 4
    LEARNING_RATE = 2e-4
    MAX_LEN = 32

# Utility Functions
def download_karpathy_json(save_path):
    if not os.path.exists(save_path):
        url = "https://github.com/Delphboy/karpathy-splits/raw/main/dataset_flickr8k.json"
        print(f"Downloading Karpathy Split JSON to {save_path}...")
        response = requests.get(url)
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print("Download complete.")

def cleanup_memory():
    """Forces garbage collection to free up VRAM between Train and Test."""
    gc.collect()
    torch.cuda.empty_cache()

# Dataset Class
class Flickr8kDataset(Dataset):
    def __init__(self, json_file, split, processor):
        with open(json_file, 'r') as f:
            data = json.load(f)
        self.images = [x for x in data['images'] if x['split'] == split]
        self.processor = processor
        self.prompt = "Write a short description for the image."
        
        # Configure tokenizer for training padding
        self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        self.processor.tokenizer.padding_side = "right"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        item = self.images[idx]
        img_path = os.path.join(Config.DATASET_DIR, item['filename'])
        
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            # Returning None lets the collate_fn drop corrupted/missing samples gracefully.
            return None 

        # Flickr8k has 5 captions; first one for training stability is picked
        caption = item['sentences'][0]['raw']
        
        # 1. Process Image + Prompt (Q-Former inputs)
        inputs = self.processor(
            images=image, 
            text=self.prompt, 
            return_tensors="pt"
        )
        
        # 2. Process Labels (LLM inputs: Prompt + Caption)
        # Concatenate prompt tokens and caption tokens manually to create targets
        prompt_tokens = self.processor.tokenizer(self.prompt, return_tensors="pt")
        caption_tokens = self.processor.tokenizer(
            caption + self.processor.tokenizer.eos_token, 
            truncation=True, 
            max_length=Config.MAX_LEN, 
            padding="max_length", 
            return_tensors="pt"
        )

        full_input_ids = torch.cat([prompt_tokens.input_ids, caption_tokens.input_ids], dim=1)
        full_attention_mask = torch.cat([prompt_tokens.attention_mask, caption_tokens.attention_mask], dim=1)
        
        # 3. Create Labels (Masking)
        labels = full_input_ids.clone()
        # Mask the prompt so model doesn't learn to generate the prompt itself
        labels[:, :prompt_tokens.input_ids.shape[1]] = -100 
        # Mask padding tokens
        labels[full_input_ids == self.processor.tokenizer.pad_token_id] = -100

        inputs["input_ids"] = full_input_ids
        inputs["attention_mask"] = full_attention_mask
        inputs["labels"] = labels
        
        return {k: v.squeeze(0) for k, v in inputs.items()}

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    # Default collator handles padding now that each sample has identical keys.
    return torch.utils.data.dataloader.default_collate(batch)

# Training Pipeline
def run_training():
    print("\n" + "="*40)
    print("TRAINING")
    print("="*40)
    
    # 1. Load Model (4-bit)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    processor = InstructBlipProcessor.from_pretrained(Config.MODEL_ID, use_fast=False)
    model = InstructBlipForConditionalGeneration.from_pretrained(
        Config.MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)

    # 2. Setup LoRA
    peft_config = LoraConfig(
        r=16, 
        lora_alpha=32, 
        lora_dropout=0.05, 
        bias="none", 
        target_modules=["q_proj", "v_proj"] 
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 3. Load Data
    train_ds = Flickr8kDataset(Config.KARPATHY_JSON, 'train', processor)

    # 4. Trainer
    args = TrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        per_device_train_batch_size=Config.BATCH_SIZE,
        gradient_accumulation_steps=Config.GRAD_ACCUMULATION,
        num_train_epochs=Config.EPOCHS,
        learning_rate=Config.LEARNING_RATE,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=collate_fn
    )

    trainer.train()
    
    print(f"Saving adapter to {Config.OUTPUT_DIR}")
    model.save_pretrained(Config.OUTPUT_DIR)
    
    # Cleanup to free VRAM for evaluation
    del model, trainer
    cleanup_memory()

# Evaluation Pipeline
def run_evaluation():
    print("\n" + "="*40)
    print("TESTING (EVALUATION)")
    print("="*40)

    # 1. Load Base Model (4-bit)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    processor = InstructBlipProcessor.from_pretrained(Config.MODEL_ID, use_fast=False)
    model = InstructBlipForConditionalGeneration.from_pretrained(
        Config.MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )

    # 2. Load Trained Adapter
    if os.path.exists(Config.OUTPUT_DIR):
        print(f"Loading adapter from {Config.OUTPUT_DIR}")
        model = PeftModel.from_pretrained(model, Config.OUTPUT_DIR)
        model.eval()
    else:
        print("Adapter not found! Evaluation cannot proceed.")
        return

    # 3. Inference Function
    def generate_caption(image):
        prompt = "Write a short description for the image."
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(Config.DEVICE)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                num_beams=5,
                max_new_tokens=20,
                min_length=5,
                repetition_penalty=1.5,
            )
        return processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()

    # 4. Run Loop
    with open(Config.KARPATHY_JSON, 'r') as f:
        data = json.load(f)
    test_images = [x for x in data['images'] if x['split'] == 'test']

    gts, res = {}, {}
    print(f"Evaluating on {len(test_images)} images")

    for item in tqdm(test_images):
        filename = item['filename']
        path = os.path.join(Config.DATASET_DIR, filename)
        if os.path.exists(path):
            try:
                image = Image.open(path).convert("RGB")
                pred = generate_caption(image)
                res[filename] = [pred]
                gts[filename] = [c['raw'] for c in item['sentences']]
            except Exception as e:
                pass

    # 5. Calculate Metrics
    if res:
        gts_subset = {k: gts[k] for k in res.keys()}
        print("\nFinal Evaluation Scores:")
        
        scorers = [
            (Bleu(4), ["Bleu-1", "Bleu-2", "Bleu-3", "Bleu-4"]),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
        ]
        
        for scorer, name in scorers:
            score, _ = scorer.compute_score(gts_subset, res)
            if isinstance(name, list):
                print(f"Bleu-4: {score[3] * 100:.2f}")
            else:
                print(f"{name}: {score * 100:.2f}")
    else:
        print("No results generated.")

if __name__ == "__main__":
    # 0. Prerequisites
    download_karpathy_json(Config.KARPATHY_JSON)
    
    if not os.path.exists(Config.DATASET_DIR):
        print(f"Dataset images not found at {Config.DATASET_DIR}")
        exit()

    # 1. Train
    run_training()
    
    # 2. Test
    run_evaluation()
