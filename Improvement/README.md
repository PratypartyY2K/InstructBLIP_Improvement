# InstructBLIP LoRA Fine-Tuning on Flickr8k

This repository contains code to evaluate a fine-tuned InstructBLIP model on the Flickr8k dataset using LoRA (Low-Rank Adaptation).

## Requirements

- Python 3.10+
- NVIDIA GPU (16GB+ VRAM recommended for 4-bit quantization)
- Java (Required for SPICE metric in COCO Eval)

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Install COCO Evaluation tools:
    ```bash
    pip install git+[https://github.com/salaniz/pycocoevalcap](https://github.com/salaniz/pycocoevalcap)
    ```

3. Ensure Java is installed (for evaluation metrics)
    ```bash
    sudo apt-get install default-jre
    ```

## Usage

Run the evaluation script pointing to your dataset and trained adapter:

```bash
python evaluate.py \
  --image_dir /path/to/flickr8k/Images \
  --caption_file /path/to/flickr8k/captions.txt \
  --adapter_path ./final_adapter
```

Run the training script
```bash
python train.py \
  --image_dir /storage/flickr8k_data/Images \
  --caption_file /storage/flickr8k_data/captions.txt \
  --output_dir ./my_model_output \
  --epochs 3
  ```

## Methodology

* **Model:** InstructBLIP (Vicuna-7b backbone)
* **Technique:** QLoRA (4-bit quantization) + PEFT
* **Metrics:** BLEU-4, CIDEr, ROUGE-L

---

### 3. `requirements.txt`
This ensures reproducibility.

```text
torch>=2.0.0
transformers>=4.31.0
peft>=0.4.0
bitsandbytes>=0.40.0
accelerate>=0.21.0
pandas
pillow
tqdm
pycocotools