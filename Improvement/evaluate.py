"""
InstructBLIP Evaluation Script for Flickr8k (Patched)
-----------------------------------------------------
Includes a 'monkey patch' for bitsandbytes to fix compatibility issues
with loading LoRA adapters that have no bias terms.
"""

import argparse
import json
import os
import torch
import pandas as pd
from PIL import Image
# Use safe tqdm to prevent widget errors
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
import bitsandbytes as bnb
from bitsandbytes.nn import Linear4bit
import warnings

warnings.filterwarnings("ignore")

# 1. PATCH for loading crashes with bitsandbytes
def safe_load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    # A. Handle Bias safely (Fixes 'NoneType' error)
    if self.bias is not None:
        bias_name = prefix + "bias"
        if bias_name in state_dict:
            bias_data = state_dict.pop(bias_name, None)
            if bias_data is not None:
                self.bias.data = bias_data.to(self.bias.data.device)

    # B. Handle Weights safely (Fixes 'KeyError')
    weight_name = prefix + "weight"
    # Only try to load weights if they exist in the file
    if weight_name in state_dict:
        self.weight, state_dict = bnb.nn.Params4bit.from_state_dict(
            state_dict, prefix=weight_name + ".", requires_grad=False
        )
    
    # C. Cleanup
    unexpected_keys.extend(state_dict.keys())

# Apply the patch immediately on import
Linear4bit._load_from_state_dict = safe_load_from_state_dict
print("bitsandbytes patch applied.")


# 2. DATASET CLASS
class Flickr8kEvalDataset(Dataset):
    def __init__(self, image_dir, caption_file, split="test"):
        self.image_dir = image_dir
        
        df = pd.read_csv(caption_file)
        df.columns = map(str.lower, df.columns)
        
        self.grouped = df.groupby('image')['caption'].apply(list).reset_index()
        
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
        references = row['caption']
        image_path = os.path.join(self.image_dir, image_name)
        return image_name, image_path, references


# 3. MAIN EVALUATION LOGIC
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    print("Loading base model...")
    model_id = "Salesforce/instructblip-vicuna-7b"
    processor = InstructBlipProcessor.from_pretrained(model_id, use_fast=False)
    
    base_model = InstructBlipForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map={"": torch.cuda.current_device()}
    )

    print(f"Loading adapter from {args.adapter_path}...")
    # The patch applied above makes this line safe
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()

    print(f"Loading dataset from {args.image_dir}...")
    eval_dataset = Flickr8kEvalDataset(args.image_dir, args.caption_file, split="test")

    results = []
    ground_truth = {}
    
    print(f"Starting evaluation on {len(eval_dataset)} images...")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Use ascii=True to prevent widget errors
    for image_name, image_path, refs in tqdm(eval_dataset, ascii=True, desc="Eval"):
        try:
            image = Image.open(image_path).convert("RGB")
            
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

            img_id = abs(hash(image_name))
            results.append({"image_id": img_id, "caption": caption})
            ground_truth[img_id] = refs
            
        except Exception as e:
            print(f"Error processing {image_name}: {e}")

    # Formatting and Saving
    coco_gen_format = results
    coco_ref_format = {
        "info": {}, "licenses": [],
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

    pred_file = os.path.join(args.output_dir, "flickr8k_preds.json")
    ref_file = os.path.join(args.output_dir, "flickr8k_refs.json")
    
    with open(pred_file, "w") as f: json.dump(coco_gen_format, f)
    with open(ref_file, "w") as f: json.dump(coco_ref_format, f)

    # Calculate Scores
    coco = COCO(ref_file)
    coco_res = coco.loadRes(pred_file)
    coco_eval = COCOEvalCap(coco, coco_res)
    coco_eval.params['image_id'] = coco_res.getImgIds()
    coco_eval.evaluate()

    print("\n" + "="*40)
    print("FINAL SCORES")
    print("="*40)
    for metric, score in coco_eval.eval.items():
        if metric in ["Bleu_4", "CIDEr", "ROUGE_L"]:
            print(f"{metric}: {score:.3f}")
    print("="*40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--caption_file", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, default="./final_adapter")
    parser.add_argument("--output_dir", type=str, default="results")
    
    args = parser.parse_args()
    main(args)