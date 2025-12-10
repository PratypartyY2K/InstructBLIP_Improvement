# InstructBLIP_Improvement Project

This repository contains two complementary workflows for the Flickr8k image-captioning dataset:

- **`Replicated/`** captures a faithful reproduction of the baseline work: zero-shot InstructBLIP evaluation and a straightforward QLoRA fine-tuning pipeline driven by the Karpathy JSON split.
- **`Improvement/`** iterates on that baseline with a CSV-based data loader, safer bitsandbytes loading, tuned decoding strategies, and separate train/eval entry points that produce COCO-compatible outputs.

Use the sections below to pick the workflow you need.

## Repository Layout

```
.
├── Replicated/
│   ├── README.md
│   ├── evaluate_zero_shot.py
│   ├── main_flickr8k.py
│   ├── requirements.txt
│   └── *.png                 # Sample result charts
└── Improvement/
    ├── README.md
    ├── train.py
    ├── evaluate.py
    ├── requirements.txt
    └── Images/ (ignored)     # Optional local copy of Flickr8k
```

Install environments independently inside each subfolder so the dependency pins remain isolated.

---

## Replicated Workflow

This folder mirrors the original experiments: a straight zero-shot benchmark plus a single-script QLoRA fine-tune + evaluation loop that operates on the official Karpathy split JSON.

### Key Scripts

- `evaluate_zero_shot.py`  
  - Downloads `dataset_flickr8k.json` if missing.  
  - Loads `Salesforce/instructblip-vicuna-7b` in full precision and captions the 1k test images with a short prompt.  
  - Reports BLEU-1..4, ROUGE-L, and CIDEr using `pycocoevalcap`.

- `main_flickr8k.py`  
  - Handles 4-bit QLoRA fine-tuning (NF4 quantization + LoRA on `q_proj`/`v_proj`) and immediately evaluates the saved adapter.  
  - Includes a `Flickr8kDataset` class that concatenates the prompt with captions and masks prompt tokens during loss computation.  
  - Computes BLEU-4, ROUGE-L, and CIDEr on the Karpathy test split.

### Running

```bash
cd Replicated
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Zero-shot evaluation
python evaluate_zero_shot.py          # update IMAGES_DIR/KARPATHY_JSON if needed

# End-to-end fine-tuning + evaluation
python main_flickr8k.py               # uses Config class for paths/hparams
```

Both scripts assume the Flickr8k Images directory exists locally and that you can download the Karpathy JSON (the helper will fetch it automatically if not present). Result plots referenced in the README are stored as PNGs inside this folder.

---

## Improvement Workflow

This folder rethinks the pipeline for easier experimentation and reproducibility.

- **Dataset ingestion** expects a CSV (`image,caption`) file, groups captions per image, and recreates the Karpathy split internally.  
- **Monkey patch for bitsandbytes** protects against missing bias tensors when loading LoRA adapters in 4-bit mode.  
- **Train/eval separation** lets you run long trainings once and re-use adapters for multiple test runs.  
- **COCO-format outputs** are saved automatically so you can re-run metrics offline later.

### Scripts

- `train.py`  
  - Loads `Salesforce/instructblip-vicuna-7b` in 4-bit NF4, attaches LoRA adapters via PEFT, and fine-tunes only the adapter weights.  
  - Uses greedy decoding with first-sentence truncation during evaluation to maximize CIDEr (~94 in prior runs).  
  - Saves predictions, references, and adapter weights to the chosen `--output_dir`.

- `evaluate.py`  
  - Shares the same dataset loader, but emphasizes qualitative inspection using beam search and longer max lengths.  
  - Generates COCO JSONs and reports BLEU-1..4, METEOR, ROUGE-L, CIDEr, and SPICE (if Java is installed).

### Running

```bash
cd Improvement
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install git+https://github.com/salaniz/pycocoevalcap   # metrics extras

# Train adapter
python train.py \
  --image_dir /path/to/Images \
  --caption_file /path/to/captions.csv \
  --output_dir ./my_adapter

# Evaluate adapter (or reuse zero-shot adapter path)
python evaluate.py \
  --image_dir /path/to/Images \
  --caption_file /path/to/captions.csv \
  --adapter_path ./my_adapter \
  --output_dir ./results
```

Refer to `Improvement/README.md` for additional tips (dataset conversion snippet, troubleshooting, etc.).

---

## Choosing a Workflow

| Use Case | Recommended Folder |
| --- | --- |
| Reproduce the original zero-shot metrics or a simple QLoRA baseline | `Replicated/` |
| Extend experiments with safer loading, CSV-based preprocessing, or COCO exports | `Improvement/` |

Both workflows leverage Hugging Face’s InstructBLIP model, so ensure you have access to Vicuna-7B weights and enough GPU memory (≥16 GB VRAM for 4-bit fine-tuning). Use `huggingface-cli login` ahead of time if the model weights are gated.
