# InstructBLIP LoRA Fine-Tuning on Flickr8k

Fine-tune Salesforce’s `instructblip-vicuna-7b` on Flickr8k using 4-bit QLoRA + PEFT and evaluate captioning quality with COCO metrics (BLEU, CIDEr, ROUGE-L, SPICE). Both scripts include a bitsandbytes monkey patch that prevents crashes when loading LoRA adapters without bias terms.

## Repository Layout

```
.
├── Images/                  # Optional local copy of Flickr8k (kept out of git)
├── evaluate.py              # Evaluation script (beam search + longer outputs)
├── train.py                 # Training script (greedy decoding tweaks)
├── requirements.txt         # Runtime dependencies
└── README.md
```

## Requirements

- Python 3.10+
- CUDA-capable GPU (≥16 GB VRAM recommended for 4-bit quantization)
- Java Runtime Environment (only needed for the SPICE metric in COCO Eval)

## Setup

```bash
pip install -r requirements.txt
pip install git+https://github.com/salaniz/pycocoevalcap   # COCO metrics
sudo apt-get install default-jre                           # Only when SPICE is needed
```

> ℹ️  The scripts expect bitsandbytes 0.40+ and transformers 4.31+. Hugging Face will download the Vicuna-7B weights on first run, so log in beforehand if the model is gated.

## Preparing Flickr8k

1. Download the Flickr8k image folder plus captions (Kaggle or the official mirror).
2. Convert the `captions.txt` file into a CSV with two columns: `image` and `caption` (each image should appear five times).
3. Place the images in `Images/` (or any folder passed to `--image_dir`).
4. Pass the CSV path to `--caption_file`. The scripts lowercase the headers and internally create the Karpathy train/val/test split (6000/1000/1000 images).

If you already have the Kaggle caption file (`Flickr8k.token.txt`), the snippet below converts it:

```python
import pandas as pd
df = pd.read_csv("Flickr8k.token.txt", sep='\t', names=["image_cap", "caption"])
df["image"] = df["image_cap"].str.split("#").str[0]
df[["image", "caption"]].to_csv("captions.csv", index=False)
```

## Training

```bash
python train.py \
  --image_dir /storage/flickr8k/Images \
  --caption_file /storage/flickr8k/captions.csv \
  --output_dir ./my_adapter \
  --epochs 3 \
  --batch_size 2 \
  --lr 1e-4
```

- `train.py` loads the Vicuna-7B backbone in 4-bit NF4 precision and fine-tunes only the LoRA adapters.
- Captions are generated with greedy decoding + early truncation to maximize CIDEr (~94+ in our runs).
- Outputs are saved inside `output_dir` (adapter weights plus COCO-format JSONs for reproducibility).

## Evaluation

```bash
python evaluate.py \
  --image_dir /path/to/flickr8k/Images \
  --caption_file /path/to/flickr8k/captions.csv \
  --adapter_path ./my_adapter \
  --output_dir ./results
```

- Uses beam search (5 beams) and longer max length to inspect qualitative outputs.
- Generates COCO-format JSON files and reports BLEU-1..4, METEOR, ROUGE-L, CIDEr, and (if Java is installed) SPICE.
- The bitsandbytes patch runs automatically, so adapters trained with or without bias tensors load cleanly.

## Tips & Troubleshooting

- Export `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128` when working on small GPUs to reduce OOM chances.
- `evaluate.py` can run on CPU for smoke tests, but the 7B model still requires significant RAM without CUDA.
- If Hugging Face authentication fails for Vicuna weights, run `huggingface-cli login` before launching either script.

## Citation

If you build on this project for academic work, please cite the original InstructBLIP paper alongside your own contribution.
