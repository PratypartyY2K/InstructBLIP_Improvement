"""
Zero-Shot Evaluation of InstructBLIP on Flickr8k (Karpathy Split)

Description:
    This script performs zero-shot image captioning using the Salesforce InstructBLIP 
    (Vicuna-7B) model. It evaluates the model on the standard Karpathy 'test' split 
    of the Flickr8k dataset using BLEU-4, ROUGE-L, and CIDEr metrics.

    Note: This script requires the 'dataset_flickr8k.json' (Karpathy split) to identify 
    the test images.

Author: Pratyush Kumar
Date: December 2025
"""

import os
import json
import torch
import requests
from PIL import Image
from tqdm import tqdm
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

# --- CONFIGURATION ---
# Path to the Flickr8k 'Images' folder
IMAGES_DIR = '/content/drive/MyDrive/flickr8k_dataset/Images' 

# Path to the Karpathy JSON split file
KARPATHY_JSON_PATH = 'dataset_flickr8k.json'

# Model Configuration
MODEL_ID = "Salesforce/instructblip-vicuna-7b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def download_karpathy_json(save_path):
    """Downloads the Karpathy split JSON if it does not exist."""
    url = "https://github.com/Delphboy/karpathy-splits/raw/main/dataset_flickr8k.json"
    print(f"Downloading Karpathy Split JSON to {save_path}...")
    response = requests.get(url)
    with open(save_path, 'wb') as f:
        f.write(response.content)
    print("Download complete.")

def load_model():
    """Loads the InstructBLIP model and processor."""
    print(f"Loading Model: {MODEL_ID}. This may take a while...")
    processor = InstructBlipProcessor.from_pretrained(MODEL_ID)
    model = InstructBlipForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to(DEVICE)
    print("Model loaded successfully.")
    return model, processor

def generate_caption(model, processor, image):
    """
    Generates a caption for a single image using zero-shot settings.
    
    Prompt: 'Write a short description for the image.' (SOTA standard for InstructBLIP)
    Constraints: Limited length to match Flickr8k style (short captions).
    """
    prompt = "Write a short description for the image."
    
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(DEVICE, torch.float16)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            num_beams=5,
            max_new_tokens=30,     # Limit verbosity
            min_length=8,
            repetition_penalty=1.5,
            length_penalty=1.0,
        )
    
    return processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()

def compute_metrics(gts, res):
    """Computes BLEU-4, ROUGE-L, and CIDEr scores."""
    print("\n--- Computing Metrics ---")
    
    scorers = [
        (Bleu(4), ["Bleu-1", "Bleu-2", "Bleu-3", "Bleu-4"]),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    
    for scorer, method in scorers:
        score, _ = scorer.compute_score(gts, res)
        if isinstance(method, list):
            for m, s in zip(method, score):
                print(f"{m}: {s*100:.2f}") # Format as percentage
        else:
            print(f"{method}: {score*100:.2f}") # Format as percentage

def main():
    # 1. Setup Data
    if not os.path.exists(KARPATHY_JSON_PATH):
        download_karpathy_json(KARPATHY_JSON_PATH)

    if not os.path.exists(IMAGES_DIR):
        print(f"Error: Images directory not found at {IMAGES_DIR}")
        return

    # 2. Parse Karpathy Split
    with open(KARPATHY_JSON_PATH, 'r') as f:
        data = json.load(f)
    
    # Filter for the 1,000 test images
    test_images = [img for img in data['images'] if img['split'] == 'test']
    print(f"Found {len(test_images)} images in Karpathy 'test' split.")

    # 3. Load Model
    model, processor = load_model()

    # 4. Inference Loop
    gts = {}  # Ground Truths
    res = {}  # Results
    
    print(f"Starting Zero-Shot Inference on {len(test_images)} images...")
    
    for item in tqdm(test_images):
        filename = item['filename']
        img_path = os.path.join(IMAGES_DIR, filename)
        
        # Skip if image file is missing locally
        if not os.path.exists(img_path):
            continue
            
        try:
            # Load and Preprocess
            image = Image.open(img_path).convert("RGB")
            
            # Generate
            pred = generate_caption(model, processor, image)
            
            # Store Result (List of 1 string)
            res[filename] = [pred]
            
            # Store Ground Truth (List of 5 strings)
            gts[filename] = [c['raw'] for c in item['sentences']]
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # 5. Evaluate
    if res:
        # Ensure comparison of successfully generated images only
        gts_subset = {k: gts[k] for k in res.keys()}
        compute_metrics(gts_subset, res)
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
