"""
Generate embeddings for preprocessed Amazon data using different PLMs.
This script reuses existing preprocessing and only generates new .feature files.
"""

import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, required=True, help='Amazon domain (e.g., All_Beauty)')
    parser.add_argument('--data_dir', type=str, default='new_processed', help='Directory with preprocessed data')
    parser.add_argument('--plm', type=str, required=True, help='HuggingFace model name (e.g., Qwen/Qwen2-7B)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for embedding generation')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--max_length', type=int, default=512, help='Max sequence length')
    parser.add_argument('--pooling', type=str, default='cls', choices=['cls', 'mean', 'last'], 
                        help='Pooling strategy: cls (first token), mean (average), last (last token)')
    
    parser.add_argument('--use_fp16', action='store_true', help='Use FP16 mixed precision')
    parser.add_argument('--use_bf16', action='store_true', help='Use BF16 mixed precision')
    parser.add_argument('--use_flash_attention', action='store_true', help='Use Flash Attention 2')
    parser.add_argument('--compile', action='store_true', help='Use torch.compile for faster inference')
    parser.add_argument('--gradient_checkpointing', action='store_true', help='Enable gradient checkpointing')
    
    return parser.parse_args()


def get_model_type(model_name):
    model_name_lower = model_name.lower()
    
    if any(x in model_name_lower for x in ['roberta', 'deberta', 'blair', 'bert', 'albert', 'minilm']):
        return 'encoder'
    elif any(x in model_name_lower for x in ['t5', 'flan']):
        return 'encoder-decoder'
    else:
        return 'decoder'


def extract_embeddings(outputs, model_type, pooling='cls', attention_mask=None):
    if model_type == 'encoder':
        if pooling == 'cls':
            return outputs.last_hidden_state[:, 0, :]
        elif pooling == 'mean':
            masked_output = outputs.last_hidden_state * attention_mask.unsqueeze(-1)
            return masked_output.sum(dim=1) / attention_mask.sum(dim=-1, keepdim=True)
    
    elif model_type == 'decoder':
        if pooling == 'last':
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(len(seq_lengths), device=seq_lengths.device)
            return outputs.last_hidden_state[batch_indices, seq_lengths, :]
        elif pooling == 'mean':
            masked_output = outputs.last_hidden_state * attention_mask.unsqueeze(-1)
            return masked_output.sum(dim=1) / attention_mask.sum(dim=-1, keepdim=True)
        elif pooling == 'cls':
            return outputs.last_hidden_state[:, 0, :]
    
    else:
        if pooling == 'mean':
            encoder_output = outputs.encoder_last_hidden_state
            masked_output = encoder_output * attention_mask.unsqueeze(-1)
            return masked_output.sum(dim=1) / attention_mask.sum(dim=-1, keepdim=True)
        else:
            return outputs.encoder_last_hidden_state[:, 0, :]


def main():
    args = parse_args()
    
    data_path = os.path.join(args.data_dir, args.domain)
    data_maps_file = os.path.join(data_path, f'{args.domain}.data_maps')
    
    if not os.path.exists(data_maps_file):
        print(f"Error: Preprocessed data not found at {data_maps_file}")
        return
    
    print(f"Found preprocessed data at {data_path}")
    
    print("Loading data maps...")
    with open(data_maps_file, 'r') as f:
        data_maps = json.load(f)
    
    print(f"Found {len(data_maps['item2id']) - 1} items")
    
    print(f"Loading model: {args.plm}")
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    dtype = torch.float32
    if args.use_bf16:
        dtype = torch.bfloat16
        print("Using BF16 mixed precision")
    elif args.use_fp16:
        dtype = torch.float16
        print("Using FP16 mixed precision")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.plm, trust_remote_code=True)
        
        model_kwargs = {
            'trust_remote_code': True,
            'torch_dtype': dtype,
            'use_safetensors': True,
        }
        
        if args.use_flash_attention:
            model_kwargs['attn_implementation'] = 'flash_attention_2'
            print("Using Flash Attention 2")
        
        model = AutoModel.from_pretrained(args.plm, **model_kwargs).to(device)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        model.eval()
        
        if args.gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled")
        
        if args.compile:
            try:
                model = torch.compile(model, mode='reduce-overhead')
                print("Model compiled with torch.compile")
            except Exception as e:
                print(f"Could not compile model: {e}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    model_type = get_model_type(args.plm)
    print(f"Model type: {model_type}")
    
    if args.pooling == 'cls' and model_type == 'decoder':
        print(f"Warning: Using 'cls' pooling with decoder model. Consider using 'last' or 'mean'")
    
    hidden_size = model.config.hidden_size
    print(f"Embedding dimension: {hidden_size}")
    
    print("Preparing item texts...")
    sorted_text = []
    
    sample_key = list(data_maps['id2meta'].keys())[0]
    use_string_keys = isinstance(sample_key, str)
    
    for i in range(1, len(data_maps['item2id'])):
        key = str(i) if use_string_keys else i
        try:
            sorted_text.append(data_maps['id2meta'][key])
        except KeyError:
            alt_key = i if use_string_keys else str(i)
            try:
                sorted_text.append(data_maps['id2meta'][alt_key])
            except KeyError:
                print(f"Error: Missing metadata for item {i}")
                raise

    
    print(f"Processing {len(sorted_text)} items...")
    
    all_embeddings = []
    
    print(f"Starting embedding generation...")
    print(f"Batch size: {args.batch_size}")
    print(f"device: {device}")
    
    with torch.no_grad():
        for pr in tqdm(range(0, len(sorted_text), args.batch_size), desc="Generating embeddings", unit="batch"):
            batch = sorted_text[pr:pr + args.batch_size]
            
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors='pt'
            ).to(device)
            
            if model_type == 'encoder-decoder':
                decoder_input_ids = torch.zeros((inputs['input_ids'].shape[0], 1), dtype=torch.long, device=device)
                outputs = model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    decoder_input_ids=decoder_input_ids
                )
            else:
                outputs = model(**inputs)
            
            embeddings = extract_embeddings(
                outputs, 
                model_type, 
                pooling=args.pooling,
                attention_mask=inputs['attention_mask']
            )
            
            all_embeddings.append(embeddings.float().cpu().numpy())
            
            if (pr // args.batch_size) % 100 == 0:
                torch.cuda.empty_cache()
    
    if len(all_embeddings) > 0:
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        
        model_name = args.plm.split('/')[-1]
        feature_file = os.path.join(data_path, f'{args.domain}.{model_name}.feature')
        
        all_embeddings.tofile(feature_file)
        
        print(f"Success! Embeddings saved to: {feature_file}")
        print(f"Shape: {all_embeddings.shape}")
    else:
        print("No embeddings generated")


if __name__ == '__main__':
    main()
