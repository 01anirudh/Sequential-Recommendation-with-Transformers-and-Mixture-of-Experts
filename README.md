# Sequential Recommendation Benchmarks

This directory contains implementation and benchmark scripts for sequential recommendation on the **Amazon Reviews 2023** dataset.

## 📋 Overview

The workflow consists of three main steps:
1.  **Data Preprocessing**: Downloading and splitting the Amazon dataset.
2.  **Embedding Generation**: Using LLMs (like Qwen, Mistral) to generate item features.
3.  **Model Training**: Training sequential recommenders (UniSRec, QwenRec, etc.) using the processed data and embeddings.

## ⚙️ Installation

Ensure you have the following dependencies installed:

```bash
pip install torch numpy pandas tqdm transformers datasets recbole
```

Optional for faster training:
```bash
pip install flash-attn --no-build-isolation
```

## 🚀 Usage Guide

### 1. Data Processing (`dataset/`)

Use `dataset/process_amazon.py` to download the Amazon Reviews 2023 dataset and process it into `train/valid/test` splits.

```bash
# Example: Process 'All_Beauty' domain
python dataset/process_amazon.py \
    --domain All_Beauty \
    --output_dir new_processed/ \
    --plm hyp1231/blair-roberta-base
```

**Arguments:**
*   `--domain`: Amazon domain name (e.g., `All_Beauty`, `Toys_and_Games`).
*   `--output_dir`: Directory to save processed data (default: `new_processed/`).
*   `--plm`: Pre-trained language model for tokenizer (default: `hyp1231/blair-roberta-base`).

### 2. Embedding Generation (`dataset/`)

If you are using LLM-based recommenders (e.g., QwenRec, MistralRec), you need to pre-compute item embeddings using `dataset/generate_embeddings_only.py`.

```bash
# Example: Generate Qwen2-7B embeddings for All_Beauty
python dataset/generate_embeddings_only.py \
    --domain All_Beauty \
    --plm Qwen/Qwen2-7B \
    --data_dir new_processed \
    --batch_size 32 \
    --use_bf16
```

**Arguments:**
*   `--plm`: HuggingFace model ID (e.g., `mistralai/Mistral-7B-v0.1`, `Qwen/Qwen2-7B`).
*   `--use_bf16` / `--use_fp16`: Use mixed precision for faster generation (Recommended).
*   `--pooling`: Pooling strategy (`cls`, `mean`, `last`).

### 3. Model Training

Use `run_with_checkpoints.py` to train and evaluate models. This script handles robust checkpointing and resuming.

```bash
# Train a QwenRec model on All_Beauty
python run_with_checkpoints.py \
    -m QwenRec \
    -d All_Beauty \
    --checkpoint_dir checkpoints
```

**Arguments:**
*   `-m`: Model name (see Supported Models below).
*   `-d`: Dataset name (must match folder in `new_processed/` or configured data path).
*   `--resume`: Resume training. Use `auto` to resume from the last checkpoint.
*   `--checkpoint_dir`: Custom directory to save model checkpoints.

## 🤖 Supported Models

The following models are supported (configurations in `config/`):

*   **UniSRec**: Universal Sequence Representation Learning
*   **QwenRec**: Sequential Recommendation with Qwen2
*   **MistralRec**: With Mistral-7B
*   **LLaMARec**: With Llama-2/3
*   **GPTRec**: With GPT-2
*   **BERT4Rec / SASRec**: Classic baselines (via RecBole)
*   **DistilRoBERTaRec**: Encoder-based recommender
*   **FlanT5SmallRec**: Encoder-Decoder recommender

## 📂 Directory Structure

*   `dataset/`: Scripts for data download, processing, and feature generation.
*   `model/`: Model implementations (inheriting from RecBole).
*   `config/`: YAML configuration files for each model.
*   `run_with_checkpoints.py`: Main training entry point.
