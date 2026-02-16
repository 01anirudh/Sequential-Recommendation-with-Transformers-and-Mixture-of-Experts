import os
import json
import numpy as np
import torch
import torch.nn as nn
from recbole.data.dataset import SequentialDataset


class UniSRecDataset(SequentialDataset):
    def __init__(self, config):
        super().__init__(config)

        self.plm_size = config['plm_size']
        self.plm_suffix = config['plm_suffix']
        plm_embedding_weight = self.load_plm_embedding()
        self.plm_embedding = self.weight2emb(plm_embedding_weight)

    def load_plm_embedding(self):
        feat_path = os.path.join(self.config['data_path'], f'{self.dataset_name}.{self.plm_suffix}')
        
        # Try multiple filename patterns to be flexible
        if not os.path.exists(feat_path):
            # If plm_suffix contains '/', try using only the part after '/'
            if '/' in self.plm_suffix:
                short_suffix = self.plm_suffix.split('/')[-1]
                alt_feat_path = os.path.join(self.config['data_path'], f'{self.dataset_name}.{short_suffix}')
                if os.path.exists(alt_feat_path):
                    feat_path = alt_feat_path
                    print(f"Using embedding file: {alt_feat_path}")
            # If plm_suffix doesn't contain '/', try adding common prefixes
            else:
                # Try with Qwen/ prefix for Qwen models
                if 'Qwen' in self.plm_suffix and not self.plm_suffix.startswith('Qwen/'):
                    alt_feat_path = os.path.join(self.config['data_path'], f'{self.dataset_name}.Qwen/{self.plm_suffix}')
                    if os.path.exists(alt_feat_path):
                        feat_path = alt_feat_path
                        print(f"Using embedding file: {alt_feat_path}")
        
        if not os.path.exists(feat_path):
            raise FileNotFoundError(
                f"Embedding file not found: {feat_path}\n"
                f"Please check that:\n"
                f"1. Embeddings were generated with generate_embeddings_only.py\n"
                f"2. The plm_suffix in config matches the actual filename\n"
                f"3. The file is in the correct directory: {self.config['data_path']}"
            )
        
        loaded_feat = np.fromfile(feat_path, dtype=np.float32).reshape(-1, self.plm_size)

        data_maps_path = os.path.join(self.config['data_path'], f'{self.dataset_name}.data_maps')
        with open(data_maps_path, 'r') as f:
            data_maps = json.load(f)

        mapped_feat = np.zeros((self.item_num, self.plm_size))
        for i, token in enumerate(self.field2id_token['item_id']):
            if token == '[PAD]': continue
            mapped_feat[i] = loaded_feat[int(data_maps['item2id'][token]) - 1]
        return mapped_feat

    def weight2emb(self, weight):
        plm_embedding = nn.Embedding(self.item_num, self.plm_size, padding_idx=0)
        plm_embedding.weight.requires_grad = False
        plm_embedding.weight.data.copy_(torch.from_numpy(weight))
        return plm_embedding


class SASRecTextDataset(UniSRecDataset):
    pass


class GPTRecDataset(UniSRecDataset):
    """Dataset for GPTRec - uses same PLM embedding loading as UniSRec"""
    pass


class RoBERTaRecDataset(UniSRecDataset):
    """Dataset for RoBERTaRec"""
    pass


class DeBERTaRecDataset(UniSRecDataset):
    """Dataset for DeBERTaRec"""
    pass


class PhiRecDataset(UniSRecDataset):
    """Dataset for PhiRec"""
    pass


class T5RecDataset(UniSRecDataset):
    """Dataset for T5Rec"""
    pass


class GemmaRecDataset(UniSRecDataset):
    """Dataset for GemmaRec"""
    pass


class QwenRecDataset(UniSRecDataset):
    """Dataset for QwenRec"""
    pass


class MistralRecDataset(UniSRecDataset):
    """Dataset for MistralRec"""
    pass


class FalconRecDataset(UniSRecDataset):
    """Dataset for FalconRec"""
    pass


class UniSRecLLMDataset(UniSRecDataset):
    """Dataset for Enhanced UniSRec with LLM embeddings"""
    pass


class UniSRecQwenDataset(UniSRecDataset):
    """Dataset for UniSRec with Qwen2-7B embeddings"""
    pass


class UniSRecMistralDataset(UniSRecDataset):
    """Dataset for UniSRec with Mistral-7B embeddings"""
    pass


class BERT4RecLLMDataset(UniSRecDataset):
    """Dataset for BERT4Rec with LLM embeddings"""
    pass


class BERT4RecQwenDataset(UniSRecDataset):
    """Dataset for BERT4Rec with Qwen2-7B embeddings"""
    pass


# Small model datasets
class FlanT5SmallRecDataset(UniSRecDataset):
    """Dataset for FlanT5SmallRec"""
    pass


class DistilRoBERTaRecDataset(UniSRecDataset):
    """Dataset for DistilRoBERTaRec"""
    pass


class MiniLMRecDataset(UniSRecDataset):
    """Dataset for MiniLMRec"""
    pass


class TinyBERTRecDataset(UniSRecDataset):
    """Dataset for TinyBERTRec"""
    pass


class ALBERTRecDataset(UniSRecDataset):
    """Dataset for ALBERTRec"""
    pass


# Better-than-BLAIR model datasets
class DeBERTaV3RecDataset(UniSRecDataset):
    """Dataset for DeBERTaV3Rec - DeBERTa-v3-Base"""
    pass


class BGERecDataset(UniSRecDataset):
    """Dataset for BGERec - BGE-Base-en-v1.5"""
    pass


class E5RecDataset(UniSRecDataset):
    """Dataset for E5Rec - E5-Base-v2"""
    pass


class LLaMARecDataset(UniSRecDataset):
    """Dataset for LLaMARec - LLaMA-3-8B"""
    pass
