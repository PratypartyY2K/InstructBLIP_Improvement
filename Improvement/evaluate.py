"""
InstructBLIP Evaluation Script for Flickr8k
-------------------------------------------
Evaluates a fine-tuned LoRA adapter for InstructBLIP on the Flickr8k dataset.
Computes BLEU-4, CIDEr, and ROUGE-L scores using standard COCO evaluation tools.

Usage:
    python evaluate.py --image_dir /path/to/images --caption_file /path/to/captions.txt --adapter_path ./final_adapter
"""

import argparse
import json
import os
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import (
    InstructBlipProcessor, 
    InstructBlipForConditionalGeneration, 
    BitsAndBytesConfig
)
from peft import PeftModel
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")


class Flickr8kEvalDataset(Dataset):
    """
    Custom Dataset for Flickr8k Evaluation.
    Returns raw image paths and reference captions (List[str]) instead of tensors.
    """
    def __init__(self, image_dir, caption_file, split="test"):
        self.image_dir = image_dir
        
        # Load and normalize CSV headers
        df = pd.read_csv(caption_file)
        df.columns = map(str.lower, df.columns) 
        
        # Group by Image (1 image : List of 5 captions)
        self.grouped = df.groupby('image')['caption'].apply(list).reset_index()
        
        # Karpathy Split Logic
        unique_images = self.grouped['image'].tolist()
        if split == "train":
            target_imgs = unique_images[:6000]
        elif split == "val":
            target_imgs = unique_images[6000:7000]
        else: # test
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


def main(args):
    # 1. Setup Device & Quantization
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    # 2. Load Base Model & Processor
    model_id = "Salesforce/instructblip-vicuna-7b"
    print(f"Loading base model: {model_id}...")
    
    # use_fast=False fixes tokenizer crash on Vicuna
    processor = InstructBlipProcessor.from_pretrained(model_id, use_fast=False)
    
    base_model = InstructBlipForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map={"": torch.cuda.current_device()} # Fixes 'different device' error
    )

    # 3. Load LoRA Adapter
    print(f"Loading LoRA adapter from {args.adapter_path}...")
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()

    # 4. Load Dataset
    print(f"Loading Test Split from {args.image_dir}...")
    eval_dataset = Flickr8kEvalDataset(args.image_dir, args.caption_file, split="test")

    # 5. Generation Loop
    results = []
    ground_truth = {}
    
    print(f"Starting evaluation on {len(eval_dataset)} images...")
    
    # Create output directory for JSONs
    os.makedirs("results", exist_ok=True)

    for image_name, image_path, refs in tqdm(eval_dataset):
        try:
            image = Image.open(image_path).convert("RGB")
            
            # Note: Prompt matches training prompt
            prompt = "Describe this image." 
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    do_sample=False,
                    num_beams=5,
                    max_length=60,
                    min_length=5,
                    repetition_penalty=1.5,
                    length_penalty=1.0,
                )
            
            caption = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()

            # Hash image name to create a unique integer ID for COCO tools
            img_id = abs(hash(image_name)) 
            
            results.append({
                "image_id": img_id,
                "caption": caption
            })
            
            ground_truth[img_id] = refs
            
        except Exception as e:
            print(f"Error processing {image_name}: {e}")

    # 6. Format for COCO Eval Tools
    coco_gen_format = results
    coco_ref_format = {
        "info": {},
        "licenses": [],
        "images": [{"id": k} for k in ground_truth.keys()],
        "annotations": []
    }

    ann_id = 0
    for img_id, refs in ground_truth.items():
        for cap in refs:
            coco_ref_format["annotations"].append({
                "image_id": img_id,
                "id": ann_id,
                "caption": cap
            })
            ann_id += 1

    # Save temp files
    pred_file = "results/flickr8k_preds.json"
    ref_file = "results/flickr8k_refs.json"
    
    with open(pred_file, "w") as f:
        json.dump(coco_gen_format, f)
    with open(ref_file, "w") as f:
        json.dump(coco_ref_format, f)

    # 7. Compute Scores
    coco = COCO(ref_file)
    coco_res = coco.loadRes(pred_file)
    
    coco_eval = COCOEvalCap(coco, coco_res)
    coco_eval.params['image_id'] = coco_res.getImgIds()
    coco_eval.evaluate()

    # 8. Print Final Metrics
    print("\n" + "="*40)
    print("FINAL SCORES (Flickr8k Test Split)")
    print("="*40)
    for metric, score in coco_eval.eval.items():
        if metric in ["Bleu_4", "CIDEr", "ROUGE_L"]:
            print(f"{metric}: {score:.3f}")
    print("="*40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate InstructBLIP on Flickr8k")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to Flickr8k images folder")
    parser.add_argument("--caption_file", type=str, required=True, help="Path to captions.txt")
    parser.add_argument("--adapter_path", type=str, default="./final_adapter", help="Path to trained LoRA adapter")
    
    args = parser.parse_args()
    main(args)