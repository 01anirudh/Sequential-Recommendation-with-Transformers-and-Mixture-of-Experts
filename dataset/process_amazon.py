import os
import re
import html
import json
import argparse
import requests
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import Dataset, DatasetDict
from transformers import AutoModel, AutoTokenizer


if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU. Please change Runtime Type if GPU is needed.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, default='All_Beauty')
    parser.add_argument('--max_his_len', type=int, default=50)
    parser.add_argument('--n_workers', type=int, default=16)
    parser.add_argument('--output_dir', type=str, default='new_processed/')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--plm', type=str, default='hyp1231/blair-roberta-base')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--min_seq_len', type=int, default=3,
                        help='Minimum user sequence length. '
                             '3 = original (full train/valid/test for all users). '
                             '2 = users with >=2 items (no train entry when len==2). '
                             '1 = all users (only test entry when len==1).')
    return parser.parse_args()


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def download_file(url, local_path):
    if os.path.exists(local_path):
        return local_path

    print(f"Downloading {url} to {local_path}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024

        with open(local_path, 'wb') as f, tqdm(
            desc=os.path.basename(local_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                size = f.write(data)
                bar.update(size)
        return local_path
    except Exception as e:
        print(f"Failed to download {url}")
        if os.path.exists(local_path):
            os.remove(local_path)
        raise e


def load_and_process_amazon_data(domain, min_seq_len=3, data_dir="data"):
    print(f"Processing Review Data for {domain}...")
    check_path(data_dir)

    file_name = f"{domain}.jsonl"
    url = f"https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/{file_name}"
    local_path = os.path.join(data_dir, file_name)
    download_file(url, local_path)

    print("Loading review data...")
    df = pd.read_json(local_path, lines=True)

    if 'timestamp' in df.columns:
        pass  # already named correctly
    elif 'sortTimestamp' in df.columns:
        df['timestamp'] = df['sortTimestamp']
    elif 'unixReviewTime' in df.columns:
        df['timestamp'] = df['unixReviewTime']
    else:
        raise KeyError(f"No timestamp column found in {domain}")

    print("Sorting and grouping interactions...")
    df = df.sort_values(by=['user_id', 'timestamp'])
    grouped = df.groupby('user_id')['parent_asin'].apply(list).reset_index()

    train_data, valid_data, test_data = [], [], []

    print(f"Splitting Data (min_seq_len={min_seq_len})...")
    skipped = 0
    for _, row in tqdm(grouped.iterrows(), total=len(grouped), desc="Splitting Data"):
        user = row['user_id']
        items = row['parent_asin']

        if len(items) < min_seq_len:
            skipped += 1
            continue
        # Test split: always possible when len >= 1
        # history must be non-empty → need len >= 2 so items[:-1] is non-empty
        if len(items) >= 2:
            test_data.append({
                'user_id': user,
                'parent_asin': items[-1],
                'history': ' '.join(items[:-1])
            })

        # Valid split: need len >= 3 so items[:-2] is non-empty (history of >=1 item)
        if len(items) >= 3:
            valid_data.append({
                'user_id': user,
                'parent_asin': items[-2],
                'history': ' '.join(items[:-2])
            })

        # Train split: need len >= 4 so items[:-3] is non-empty (history of >=1 item)
        if len(items) >= 4:
            train_data.append({
                'user_id': user,
                'parent_asin': items[-3],
                'history': ' '.join(items[:-3])
            })

    print(f"Skipped {skipped} users with < {min_seq_len} interactions")
    print(f"Split sizes → train: {len(train_data)}, valid: {len(valid_data)}, test: {len(test_data)}")

    return DatasetDict({
        'train': Dataset.from_list(train_data),
        'valid': Dataset.from_list(valid_data),
        'test': Dataset.from_list(test_data)
    })


def process_meta(args, data_dir="data"):
    domain = args.domain
    print(f"Processing Metadata for {domain}...")
    check_path(data_dir)

    if domain.startswith("meta_"):
        file_name = f"{domain}.jsonl"
    else:
        file_name = f"meta_{domain}.jsonl"

    meta_url = f"https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories/{file_name}"
    local_path = os.path.join(data_dir, file_name)
    download_file(meta_url, local_path)

    print("Loading metadata...")
    df_meta = pd.read_json(local_path, lines=True)

    item2meta = {}
    features_needed = ['title', 'features', 'categories', 'description']

    print("Cleaning metadata...")
    for _, row in tqdm(df_meta.iterrows(), total=len(df_meta)):
        if 'parent_asin' not in row:
            continue

        parent_asin = row['parent_asin']
        meta_text = ''

        for feature in features_needed:
            if feature not in row:
                continue

            val = row[feature]

            if val is None:
                continue

            if isinstance(val, list):
                if len(val) > 0:
                    meta_text += feature_process(val)
            elif pd.notnull(val):
                meta_text += feature_process(val)

        item2meta[parent_asin] = meta_text

    return item2meta


def list_to_str(l):
    if isinstance(l, list):
        return list_to_str(', '.join(l))
    return l


def clean_text(raw_text):
    if raw_text is None:
        return ""
    text = list_to_str(raw_text)
    text = html.unescape(text)
    text = text.strip()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[\n\t]', ' ', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'[^\x00-\x7F]', ' ', text)
    return text


def feature_process(feature):
    sentence = ""
    if isinstance(feature, float):
        sentence += str(feature) + '.'
    elif isinstance(feature, list) and len(feature) > 0:
        for v in feature:
            sentence += clean_text(v) + ', '
        sentence = sentence[:-2] + '.'
    else:
        sentence = clean_text(feature)
    return sentence + ' '


def filter_items_wo_metadata(example, item2meta):
    """Remove items without metadata from history. If target has no meta, clear history."""
    if example['parent_asin'] not in item2meta:
        example['history'] = ''
    history_items = example['history'].split(' ')
    filtered = [item for item in history_items if item in item2meta]
    example['history'] = ' '.join(filtered)
    return example


def truncate_history(example, max_his_len):
    example['history'] = ' '.join(example['history'].split(' ')[-max_his_len:])
    return example


def remap_id(datasets):
    """Map user/item ASINs to integer IDs. Guards against empty-string items."""
    user2id = {'[PAD]': 0}
    id2user = ['[PAD]']
    item2id = {'[PAD]': 0}
    id2item = ['[PAD]']

    for split in ['train', 'valid', 'test']:
        dataset = datasets[split]
        for user_id, item_id, history in zip(
            dataset['user_id'], dataset['parent_asin'], dataset['history']
        ):
            if user_id not in user2id:
                user2id[user_id] = len(id2user)
                id2user.append(user_id)
            if item_id not in item2id:
                item2id[item_id] = len(id2item)
                id2item.append(item_id)

            # BUG FIX: ''.split(' ') == [''] — skip the empty string token
            history_items = history.split(' ') if history.strip() else []
            for item in history_items:
                if item and item not in item2id:   # `item` check skips empty string
                    item2id[item] = len(id2item)
                    id2item.append(item)

    return {
        'user2id': user2id,
        'id2user': id2user,
        'item2id': item2id,
        'id2item': id2item
    }


if __name__ == '__main__':
    args = parse_args()

    datasets = load_and_process_amazon_data(args.domain, min_seq_len=args.min_seq_len)
    item2meta = process_meta(args)

    truncated_datasets = {}
    output_dir = os.path.join(args.output_dir, args.domain)
    check_path(output_dir)

    for split in ['train', 'valid', 'test']:
        print(f"Filtering {split} split...")

        # Remove items without metadata from history
        filtered_dataset = datasets[split].map(
            lambda t: filter_items_wo_metadata(t, item2meta),
            num_proc=args.n_workers
        )

        # Keep only rows where:
        # 1. Target item has metadata → can be embedded
        # 2. History is non-empty   → item_seq_len > 0 (avoids gather_indexes(-1) CUDA crash)
        filtered_dataset = filtered_dataset.filter(
            lambda t: t['parent_asin'] in item2meta and len(t['history'].strip()) > 0
        )

        truncated_dataset = filtered_dataset.map(
            lambda t: truncate_history(t, args.max_his_len),
            num_proc=args.n_workers
        )
        truncated_datasets[split] = truncated_dataset

        output_path = os.path.join(output_dir, f'{args.domain}.{split}.inter')
        with open(output_path, 'w') as f:
            f.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
            for user_id, history, parent_asin in zip(
                truncated_dataset['user_id'],
                truncated_dataset['history'],
                truncated_dataset['parent_asin']
            ):
                f.write(f"{user_id}\t{history}\t{parent_asin}\n")

        print(f"  {split}: {len(truncated_dataset)} interactions written")

    print("Remapping IDs...")
    data_maps = remap_id(truncated_datasets)

    # Build id2meta: only include items that actually have metadata.
    # Items in item2id without metadata should NOT appear in the dataset
    # (filtered above), but we guard anyway.
    id2meta = {0: '[PAD]'}
    missing_meta = 0
    for asin, item_id in data_maps['item2id'].items():
        if asin == '[PAD]':
            continue
        if asin in item2meta:
            id2meta[item_id] = item2meta[asin]
        else:
            id2meta[item_id] = ''   # safety fallback — should not happen given filters above
            missing_meta += 1

    if missing_meta > 0:
        print(f"WARNING: {missing_meta} items in item2id have no metadata (empty embedding fallback)")

    data_maps['id2meta'] = id2meta

    output_path = os.path.join(output_dir, f'{args.domain}.data_maps')
    with open(output_path, 'w') as f:
        json.dump(data_maps, f)

    print(f"\nFinal Statistics:")
    print(f"  #Users : {len(data_maps['user2id']) - 1}")
    print(f"  #Items : {len(data_maps['item2id']) - 1}")
    n_interactions = {split: len(truncated_datasets[split]) for split in ['train', 'valid', 'test']}
    print(f"  #Interactions : {sum(n_interactions.values())}")
    print(f"  Per split     : {n_interactions}")
