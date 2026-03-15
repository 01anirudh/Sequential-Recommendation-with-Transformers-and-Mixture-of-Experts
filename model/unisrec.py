"""
UniSRecImproved — Enhanced UniSRec Sequential Recommender

Improvements over the original UniSRec:
1. Enhanced MoE Adapter  : two-stage projection (plm → 2×hidden → hidden) + residual
2. Layer-wise aggregation : learnable weighted sum over all SASRec transformer layers
3. Learnable temperature  : nn.Parameter instead of a fixed float
4. Label smoothing        : CrossEntropyLoss(label_smoothing=0.1) for fine-tuning
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.sequential_recommender.sasrec import SASRec


# ──────────────────────────────────────────────────────────────────────────────
# 1.  MoE Adapter (enhanced)
# ──────────────────────────────────────────────────────────────────────────────

class PWLayer(nn.Module):
    """Parametric Whitening Layer — single expert."""

    def __init__(self, input_size, output_size, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.bias    = nn.Parameter(torch.zeros(input_size), requires_grad=True)
        self.lin     = nn.Linear(input_size, output_size, bias=False)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.lin.weight, mean=0.0, std=0.02)

    def forward(self, x):
        return self.lin(self.dropout(x) - self.bias)


class DeepPWLayer(nn.Module):
    """Two-stage expert: plm_size → mid → output, with an optional residual.

    Architecture:
        x → dropout → (x - bias_in) → lin_in → ReLU → lin_out → + shortcut(x)
    """

    def __init__(self, input_size, output_size, dropout=0.0, expansion=2):
        super().__init__()
        mid = input_size * expansion
        self.dropout  = nn.Dropout(p=dropout)
        self.bias_in  = nn.Parameter(torch.zeros(input_size), requires_grad=True)
        self.lin_in   = nn.Linear(input_size, mid, bias=False)
        self.act      = nn.GELU()
        self.lin_out  = nn.Linear(mid, output_size, bias=True)

        # Residual shortcut when dims differ
        self.shortcut = (
            nn.Linear(input_size, output_size, bias=False)
            if input_size != output_size
            else nn.Identity()
        )
        self._init_weights()

    def _init_weights(self):
        for m in [self.lin_in, self.lin_out]:
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if hasattr(self.shortcut, 'weight'):
            nn.init.normal_(self.shortcut.weight, mean=0.0, std=0.02)

    def forward(self, x):
        h = self.lin_in(self.dropout(x) - self.bias_in)
        h = self.act(h)
        h = self.lin_out(h)
        return h + self.shortcut(x)          # residual connection


class MoEAdaptorLayerImproved(nn.Module):
    """MoE adapter with deep experts and noisy top-k gating."""

    def __init__(self, n_exps, layers, dropout=0.0, noise=True):
        super().__init__()
        self.n_exps       = n_exps
        self.noisy_gating = noise

        input_size, output_size = layers[0], layers[1]
        self.experts = nn.ModuleList(
            [DeepPWLayer(input_size, output_size, dropout) for _ in range(n_exps)]
        )
        self.w_gate  = nn.Parameter(torch.zeros(input_size, n_exps), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(input_size, n_exps), requires_grad=True)

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise        = x @ self.w_noise
            noise_stddev     = F.softplus(raw_noise) + noise_epsilon
            clean_logits     = clean_logits + torch.randn_like(clean_logits) * noise_stddev
        return F.softmax(clean_logits, dim=-1)   # (B, n_exps)

    def forward(self, x):
        gates  = self.noisy_top_k_gating(x, self.training)          # (B, n_exps)
        outs   = torch.stack([e(x) for e in self.experts], dim=-2)   # (B, n_exps, D)
        return (gates.unsqueeze(-1) * outs).sum(dim=-2)              # (B, D)


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Main model
# ──────────────────────────────────────────────────────────────────────────────

class UniSRecImproved(SASRec):
    """
    Improved UniSRec with:
      • Deep MoE adapter (2-stage + residual)
      • Learnable weighted aggregation of all transformer layers
      • Learnable temperature
      • Label smoothing in fine-tuning CE loss
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.train_stage = config['train_stage']
        self.lam         = config['lambda']
        try:
            self.label_smoothing = config['label_smoothing']
        except (KeyError, AttributeError):
            self.label_smoothing = 0.1

        # Learnable temperature (initialised from config)
        init_temp = float(config['temperature'])
        self.log_temperature = nn.Parameter(torch.tensor([init_temp]).log())

        assert self.train_stage in ['pretrain', 'inductive_ft', 'transductive_ft'], \
            f'Unknown train stage: [{self.train_stage}]'

        if self.train_stage in ['pretrain', 'inductive_ft']:
            self.item_embedding = None
        if self.train_stage in ['inductive_ft', 'transductive_ft']:
            self.plm_embedding = copy.deepcopy(dataset.plm_embedding)

        # Enhanced MoE adapter
        self.moe_adaptor = MoEAdaptorLayerImproved(
            config['n_exps'],
            config['adaptor_layers'],
            config['adaptor_dropout_prob'],
        )

        # Learnable layer weights for aggregating all transformer outputs
        n_layers = config['n_layers']
        self.layer_weights = nn.Parameter(torch.ones(n_layers) / n_layers)

        # Override CE loss with label smoothing (only affects fine-tuning)
        self.loss_fct = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

    # ── helpers ───────────────────────────────────────────────────────────────

    @property
    def temperature(self):
        # clamp to [0.001, 5.0] to stay numerically sane
        return self.log_temperature.exp().clamp(1e-3, 5.0)

    def _aggregate_layers(self, all_layer_outputs):
        """Weighted mean over all transformer-layer outputs.

        all_layer_outputs : list of (B, L, H)  length = n_layers
        returns            : (B, L, H)
        """
        stacked = torch.stack(all_layer_outputs, dim=0)           # (n_layers, B, L, H)
        weights = F.softmax(self.layer_weights, dim=0)            # (n_layers,)
        return (weights[:, None, None, None] * stacked).sum(0)    # (B, L, H)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, item_seq, item_emb, item_seq_len):
        position_ids      = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids      = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        input_emb = item_emb + position_embedding
        if self.train_stage == 'transductive_ft':
            input_emb = input_emb + self.item_embedding(item_seq)
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)
        all_layers = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)

        # ─── layer-wise aggregation ───
        output = self._aggregate_layers(all_layers)              # (B, L, H)
        output = self.gather_indexes(output, item_seq_len - 1)   # (B, H)
        return output

    # ── contrastive tasks (same logic, use self.temperature property) ─────────

    def seq_item_contrastive_task(self, seq_output, same_pos_id, interaction):
        pos_items_emb = self.moe_adaptor(interaction['pos_item_emb'])
        pos_items_emb = F.normalize(pos_items_emb, dim=1)

        pos_logits = (seq_output * pos_items_emb).sum(dim=1) / self.temperature
        pos_logits = torch.exp(pos_logits)

        neg_logits = torch.matmul(seq_output, pos_items_emb.T) / self.temperature
        neg_logits = torch.where(
            same_pos_id,
            torch.zeros_like(neg_logits),
            neg_logits,
        )
        neg_logits = torch.exp(neg_logits).sum(dim=1)

        return (-torch.log(pos_logits / neg_logits)).mean()

    def seq_seq_contrastive_task(self, seq_output, same_pos_id, interaction):
        item_seq_aug     = interaction[self.ITEM_SEQ + '_aug']
        item_seq_len_aug = interaction[self.ITEM_SEQ_LEN + '_aug']
        item_emb_aug     = self.moe_adaptor(interaction['item_emb_list_aug'])
        seq_output_aug   = self.forward(item_seq_aug, item_emb_aug, item_seq_len_aug)
        seq_output_aug   = F.normalize(seq_output_aug, dim=1)

        pos_logits = (seq_output * seq_output_aug).sum(dim=1) / self.temperature
        pos_logits = torch.exp(pos_logits)

        neg_logits = torch.matmul(seq_output, seq_output_aug.T) / self.temperature
        neg_logits = torch.where(
            same_pos_id,
            torch.zeros_like(neg_logits),
            neg_logits,
        )
        neg_logits = torch.exp(neg_logits).sum(dim=1)

        return (-torch.log(pos_logits / neg_logits)).mean()

    # ── pretrain / fine-tune ─────────────────────────────────────────────────

    def pretrain(self, interaction):
        item_seq      = interaction[self.ITEM_SEQ]
        item_seq_len  = interaction[self.ITEM_SEQ_LEN]
        item_emb_list = self.moe_adaptor(interaction['item_emb_list'])
        seq_output    = self.forward(item_seq, item_emb_list, item_seq_len)
        seq_output    = F.normalize(seq_output, dim=1)

        pos_id      = interaction['item_id']
        same_pos_id = pos_id.unsqueeze(1) == pos_id.unsqueeze(0)
        same_pos_id = torch.logical_xor(same_pos_id, torch.eye(
            pos_id.shape[0], dtype=torch.bool, device=pos_id.device))

        loss = (
            self.seq_item_contrastive_task(seq_output, same_pos_id, interaction)
            + self.lam * self.seq_seq_contrastive_task(seq_output, same_pos_id, interaction)
        )
        return loss

    def calculate_loss(self, interaction):
        if self.train_stage == 'pretrain':
            return self.pretrain(interaction)

        item_seq      = interaction[self.ITEM_SEQ]
        item_seq_len  = interaction[self.ITEM_SEQ_LEN]
        item_emb_list = self.moe_adaptor(self.plm_embedding(item_seq))
        seq_output    = self.forward(item_seq, item_emb_list, item_seq_len)

        test_item_emb = self.moe_adaptor(self.plm_embedding.weight)
        if self.train_stage == 'transductive_ft':
            test_item_emb = test_item_emb + self.item_embedding.weight

        seq_output    = F.normalize(seq_output, dim=1)
        test_item_emb = F.normalize(test_item_emb, dim=1)

        logits    = torch.matmul(seq_output, test_item_emb.T) / self.temperature
        pos_items = interaction[self.POS_ITEM_ID]
        return self.loss_fct(logits, pos_items)

    def full_sort_predict(self, interaction):
        item_seq      = interaction[self.ITEM_SEQ]
        item_seq_len  = interaction[self.ITEM_SEQ_LEN]
        item_emb_list = self.moe_adaptor(self.plm_embedding(item_seq))
        seq_output    = self.forward(item_seq, item_emb_list, item_seq_len)

        test_items_emb = self.moe_adaptor(self.plm_embedding.weight)
        if self.train_stage == 'transductive_ft':
            test_items_emb = test_items_emb + self.item_embedding.weight

        seq_output     = F.normalize(seq_output, dim=-1)
        test_items_emb = F.normalize(test_items_emb, dim=-1)

        return torch.matmul(seq_output, test_items_emb.T)   # (B, n_items)
