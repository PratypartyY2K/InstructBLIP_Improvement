# InstructBLIP Flickr8k Toolkit

This repository now ships a single, flat project layout with four runnable entry points for experimenting with Salesforce’s `instructblip-vicuna-7b` on the Flickr8k dataset. The previous `Replicated/` and `Improvement/` folders have been merged so that all scripts, requirements, and documentation live at the root of the repo. Result figures have been consolidated under `images/`.

## Scripts at a Glance

| File | What it does |
| --- | --- |
| `zero_shot_caption_eval.py` | Runs the pure zero-shot benchmark on the Karpathy Flickr8k test split and reports BLEU-1..4, ROUGE-L, and CIDEr. |
| `qlora_train_and_eval.py` | Fine-tunes InstructBLIP with QLoRA (NF4 quantization + LoRA adapters) on the Karpathy train JSON and evaluates the saved adapter on the test split. |
| `greedy_adapter_eval.py` | Loads a previously trained adapter and evaluates it with greedy decoding plus first-sentence truncation; optimized for high CIDEr scores. |
| `beam_adapter_eval.py` | Adapter evaluation with beam search and longer outputs for qualitative inspection; writes COCO-format prediction/reference files. |
| `qualitative_eval.py` | Convenience viewer that plots selected Flickr8k images with predicted captions alongside a reference caption. |

All PNG assets referenced in prior reports are now in `images/`.

## Environment Setup

1. Create and activate a Python 3.10+ virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install git+https://github.com/salaniz/pycocoevalcap  # optional, for SPICE/METEOR extras
   ```
3. Ensure you can download the gated Vicuna weights:
   ```bash
   huggingface-cli login
   ```
4. Install `default-jre` if you need the SPICE metric from COCO Eval.

## Preparing Flickr8k Assets

- Download the Flickr8k `Images` directory plus caption file from Kaggle or another mirror.
- Convert `Flickr8k.token.txt` into a CSV with two columns (`image,caption`). Example:
  ```python
  import pandas as pd
  df = pd.read_csv("Flickr8k.token.txt", sep="\t", names=["image_cap", "caption"])
  df["image"] = df["image_cap"].str.split("#").str[0]
  df[["image", "caption"]].to_csv("captions.csv", index=False)
  ```
- Store the Karpathy JSON split (`dataset_flickr8k.json`) next to the scripts or let them auto-download it.
- Update command-line arguments or constants with the paths to your `Images/` directory and CSV/JSON files.

## Usage

### Zero-Shot Baseline

Edit `IMAGES_DIR` and `KARPATHY_JSON_PATH` at the top of `zero_shot_caption_eval.py`, then run:

```bash
python zero_shot_caption_eval.py
```

The script downloads the Karpathy split if missing, loads the model in full precision, and reports BLEU, ROUGE-L, and CIDEr for the 1,000-image test set.

### End-to-End QLoRA Fine-Tuning

Update the values inside the `Config` class in `qlora_train_and_eval.py` (paths, epochs, batch size, etc.), then run:

```bash
python qlora_train_and_eval.py
```

Key characteristics:
- Loads InstructBLIP in 4-bit NF4 using bitsandbytes.
- Applies LoRA on `q_proj` and `v_proj` layers only.
- Masks the prompt tokens in the loss to focus learning on the caption portion.
- Saves adapters plus intermediate checkpoints to `--output_dir`, then reloads them for evaluation.

### Adapter Evaluations with CSV Input

Both adapter evaluation scripts expect a CSV containing five captions per image and internally rebuild the Karpathy split (6000/1000/1000).

Greedy decoding (CIDEr-oriented):

```bash
python greedy_adapter_eval.py \
  --image_dir /path/to/Images \
  --caption_file ./captions.csv \
  --adapter_path ./adapters/flickr8k_lora \
  --output_dir ./results_greedy
```

Beam-search decoding (qualitative/long-form):

```bash
python beam_adapter_eval.py \
  --image_dir /path/to/Images \
  --caption_file ./captions.csv \
  --adapter_path ./adapters/flickr8k_lora \
  --output_dir ./results_beam
```

Both scripts:
- Apply a safe bitsandbytes monkey patch so adapters trained without bias terms can be reloaded.
- Emit COCO-style JSONs (`flickr8k_preds.json`, `flickr8k_refs.json`) for offline metric calculations.
- Report BLEU, ROUGE-L, CIDEr, and (optionally) SPICE when Java is available.

### Quick Qualitative Inspection

When you only need a few spot checks, point `qualitative_eval.py` to your Flickr8k assets, update `adapter_path`, and edit `indices_to_show`:

```bash
python qualitative_eval.py
```

The script will pop up a Matplotlib figure with the chosen images while printing their predicted captions and a sample ground-truth caption to the console.

## Tips & Troubleshooting

- Set `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128` on smaller GPUs to reduce fragmentation.
- For purely CPU smoke tests, lower `max_length`, `num_beams`, and batch sizes to keep memory manageable.
- If the Vicuna checkpoint download fails, double-check your Hugging Face credentials and model access.
- All project figures live under `images/`; keep the folder if you need to embed them into reports or slides.

Happy captioning!
