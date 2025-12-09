# InstructBLIP_Improvement

A small project for evaluating and improving InstructBLIP in a zero-shot setting.

## Overview
This repository contains scripts to run zero-shot evaluation of an InstructBLIP model and a helper script for experiments on the Flickr8k dataset.

Files of interest:
- `evaluate_zero_shot.py` — runs zero-shot evaluation.
- `main_flickr8k.py` — example / experiment driver for Flickr8k.

## Prerequisites
- Python 3.8+ (tested on macOS)
- CUDA and a compatible GPU if running heavy model inference (optional but recommended)

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Running the zero-shot evaluation
After installing dependencies, run:

```bash
python evaluate_zero_shot.py
```

This will execute the zero-shot evaluation routine. Check the top of `evaluate_zero_shot.py` for configurable options (paths, model names, device selection).

## Running Flickr8k experiments
To run the example driver (Flickr8k) use:

```bash
python main_flickr8k.py
```

Inspect `main_flickr8k.py` for dataset paths and other parameters to configure experiments.

## Troubleshooting
- If you see CUDA/torch device errors, ensure the installed PyTorch build matches your CUDA version or switch to CPU by setting the device in the scripts.
- If packages are missing, re-run `pip install -r requirements.txt` and check for any error messages.

## License & Acknowledgements
This repository is for academic experimentation. See individual scripts for references and authorship.