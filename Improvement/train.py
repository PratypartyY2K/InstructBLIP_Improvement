"""
InstructBLIP LoRA Fine-Tuning Script for Flickr8k
-------------------------------------------------
Fine-tunes InstructBLIP (Vicuna-7b backbone) on the Flickr8k dataset using QLoRA.
Optimized for 16GB VRAM GPUs (e.g., NVIDIA A4000).

Features:
- 4-bit Quantization (QLoRA)
- Memory Optimization (Batch Size 1 + Gradient Accumulation 16)
- Custom Data Collator for Multimodal Inputs
- Proper Training/Validation Splits
"""

import argparse
import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    InstructBlipProcessor, 
    InstructBlipForConditionalGeneration, 
    TrainingArguments, 
    Trainer,
    BitsAndBytesConfig
)
from peft import (
    get_peft_model, 
    LoraConfig, 
    prepare_model_for_kbit_training
)

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")


class Flickr8kTrainDataset(Dataset):
    """
    Dataset loader for Flickr8k that prepares inputs for InstructBLIP training.
    Splits data based on unique image IDs to prevent leakage.
    """
    def __init__(self, image_dir, caption_file, processor, split="train"):
        self.image_dir = image_dir
        self.processor = processor
        
        # Load Data
        df = pd.read_csv(caption_file)
        df.columns = map(str.lower, df.columns)
        
        # Split Logic (Karpathy Split: 6000 Train / 1000 Val / 1000 Test)
        unique_images = df['image'].unique().tolist()
        
        if split == "train":
            target_images = unique_images[:6000]
        elif split == "val":
            target_images = unique_images[6000:7000]
        else:
            target_images = unique_images[7000:]
            
        self.df = df[df['image'].isin(target_images)].reset_index(drop=True)
        print(f"Loaded {split} split: {len(self.df)} samples.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_filename = row['image']
        caption = row['caption']
        
        image_path = os.path.join(self.image_dir, image_filename)
        image = Image.open(image_path).convert("RGB")
        
        # 1. Q-Former Input (The Prompt)
        inputs = self.processor(
            images=image, 
            text="Describe this image.", 
            return_tensors="pt"
        )
        
        # 2. LLM Input (The Target Caption)
        caption_inputs = self.processor(text=caption, return_tensors="pt")
        
        # Align inputs and labels
        inputs['input_ids'] = caption_inputs['input_ids']
        inputs['attention_mask'] = caption_inputs['attention_mask']
        inputs['labels'] = caption_inputs['input_ids']
        
        return {k: v.squeeze() for k, v in inputs.items()}


class InstructBlipCollator:
    """
    Custom collator to handle padding for multimodality (Images + Q-Former Text + LLM Text).
    Masks padding tokens with -100 so the loss function ignores them.
    """
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        pixel_values = torch.stack([x['pixel_values'] for x in batch])
        input_ids = [x['input_ids'] for x in batch]
        labels = [x['labels'] for x in batch]
        qformer_input_ids = [x['qformer_input_ids'] for x in batch]
        qformer_attention_mask = [x['qformer_attention_mask'] for x in batch]
        
        # Pad Inputs
        padded_inputs = self.processor.tokenizer.pad(
            {'input_ids': input_ids}, 
            padding=True, 
            return_tensors="pt"
        )
        
        # Pad Labels & Mask Padding
        padded_labels = self.processor.tokenizer.pad(
            {'input_ids': labels}, 
            padding=True, 
            return_tensors="pt"
        )
        labels = padded_labels['input_ids']
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        # Pad Q-Former
        padded_qformer = self.processor.tokenizer.pad(
            {'input_ids': qformer_input_ids, 'attention_mask': qformer_attention_mask},
            padding=True,
            return_tensors="pt"
        )
        
        return {
            'pixel_values': pixel_values,
            'input_ids': padded_inputs['input_ids'],
            'attention_mask': padded_inputs['attention_mask'],
            'labels': labels,
            'qformer_input_ids': padded_qformer['input_ids'],
            'qformer_attention_mask': padded_qformer['attention_mask']
        }


def main(args):
    # 1. Setup & Config
    device_index = torch.cuda.current_device()
    print(f"Using GPU Device Index: {device_index}")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    # 2. Load Processor & Model
    model_id = "Salesforce/instructblip-vicuna-7b"
    
    # use_fast=False fixes known Rust tokenizer crashes
    processor = InstructBlipProcessor.from_pretrained(model_id, use_fast=False)

    model = InstructBlipForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map={"": device_index}
    )

    # Prepare for LoRA
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # 3. LoRA Configuration
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none"
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 4. Datasets
    train_dataset = Flickr8kTrainDataset(args.image_dir, args.caption_file, processor, split="train")
    # Optional: You can uncomment validation if you want to evaluate during training
    # val_dataset = Flickr8kTrainDataset(args.image_dir, args.caption_file, processor, split="val")

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        logging_steps=20,
        save_strategy="epoch",
        fp16=True,
        remove_unused_columns=False,
        report_to="none"
    )

    # 6. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=InstructBlipCollator(processor)
    )

    # 7. Start Training
    print("Starting Training...")
    trainer.train()

    # 8. Save Adapter
    final_path = os.path.join(args.output_dir, "final_adapter")
    model.save_pretrained(final_path)
    print(f"Training Complete. Adapter saved to {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune InstructBLIP on Flickr8k")
    
    # Paths
    parser.add_argument("--image_dir", type=str, required=True, help="Path to images")
    parser.add_argument("--caption_file", type=str, required=True, help="Path to captions.txt")
    parser.add_argument("--output_dir", type=str, default="./instructblip-flickr8k", help="Output directory")
    
    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--grad_accum", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    
    args = parser.parse_args()
    main(args)