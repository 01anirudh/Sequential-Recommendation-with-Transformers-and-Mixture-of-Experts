"""
DeBERTa-v3 Sequential Recommender
Based on UniSRec architecture with DeBERTa-v3 embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.sequential_recommender.sasrec import SASRec


class DeBERTaV3Rec(SASRec):
    """
    DeBERTa-v3-Base based sequential recommender.
    Uses pre-computed DeBERTa-v3 embeddings with a simple adapter layer.
    """
    
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        
        # Remove the default item embedding (we use PLM embeddings instead)
        self.item_embedding = None
        
        # Load pre-computed PLM embeddings from dataset
        self.plm_embedding = dataset.plm_embedding
        
        # Configuration
        self.temperature = config['temperature']
        self.plm_size = config['plm_size']
        
        # Simple adapter to project PLM embeddings to hidden_size
        self.adaptor = nn.Linear(self.plm_size, self.hidden_size)
        self.adaptor_dropout = nn.Dropout(config['adaptor_dropout_prob'])
        
    def forward(self, item_seq, item_seq_len):
        """Forward pass using PLM embeddings"""
        # Get PLM embeddings for the sequence
        item_emb = self.plm_embedding(item_seq)  # [B, L, plm_size]
        
        # Apply adapter
        item_emb = self.adaptor(self.adaptor_dropout(item_emb))  # [B, L, hidden_size]
        
        # Add position embeddings
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)
        
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        
        # Pass through transformer
        extended_attention_mask = self.get_attention_mask(item_seq)
        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        
        # Get final representation
        output = self.gather_indexes(output, item_seq_len - 1)  # [B, H]
        return output
    
    def calculate_loss(self, interaction):
        """Calculate contrastive loss"""
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        
        # Get sequence representation
        seq_output = self.forward(item_seq, item_seq_len)
        seq_output = F.normalize(seq_output, dim=1)
        
        # Get all item embeddings
        test_item_emb = self.adaptor(self.plm_embedding.weight)  # [n_items, hidden_size]
        test_item_emb = F.normalize(test_item_emb, dim=1)
        
        # Calculate logits
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1)) / self.temperature
        
        # Cross-entropy loss
        pos_items = interaction[self.POS_ITEM_ID]
        loss = self.loss_fct(logits, pos_items)
        
        return loss
    
    def full_sort_predict(self, interaction):
        """Predict scores for all items"""
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        
        # Get sequence representation
        seq_output = self.forward(item_seq, item_seq_len)
        seq_output = F.normalize(seq_output, dim=-1)
        
        # Get all item embeddings
        test_items_emb = self.adaptor(self.plm_embedding.weight)
        test_items_emb = F.normalize(test_items_emb, dim=-1)
        
        # Calculate scores
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        
        return scores
