"""DUPGT model with safe handling for empty rating histories."""
from __future__ import annotations

import torch

from .upstream import DeepMetaNet, Model


# Keep empty Book histories from acting like ratings.
class SafeDeepMetaNet(DeepMetaNet):
    """Same attention/decoder weights; padding never becomes observed evidence.

    Upstream softmax assigns uniform attention when every position is padding.
    An empty branch here receives a zero history summary before its (unchanged)
    decoder. The decoder may still learn a bias; there is no added no-transfer
    branch, count confidence, semantics, or research extension in this baseline.
    """

    def forward(self, emb_fea, seq_index):
        present = seq_index.ne(0).unsqueeze(-1)
        logits = self.event_K(emb_fea).masked_fill(~present, -1e8)
        attention = self.event_softmax(logits) * present
        attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1e-12)
        summary = torch.sum(attention * emb_fea, dim=1)
        return self.decoder(summary)


# Turn Book histories into Movie rating predictions.
class DUPGTModel(Model):
    def __init__(self, n_users: int, n_items: int, embedding_dim: int):
        super().__init__(n_users, n_items, 2, embedding_dim)
        # Preserve initialization and every parameter name from the upstream file.
        for name in ["bridging_high", "bridging_low"]:
            old = getattr(self.dupgt, name)
            safe = SafeDeepMetaNet(embedding_dim)
            safe.load_state_dict(old.state_dict())
            setattr(self.dupgt, name, safe)

    # Mix high and low Book paths for this Movie.
    def forward_with_gate(self, x):
        source_user = self.src_model.uid_embedding(x[:, 0]).unsqueeze(1)
        target_item = self.tgt_model.iid_embedding(x[:, 1])
        sequence = x[:, 2:42]
        features = self.src_model.iid_embedding(sequence)
        high = self.dupgt.bridging_high(features[:, :20], sequence[:, :20])
        low = self.dupgt.bridging_low(features[:, 20:], sequence[:, 20:])
        alpha = self.dupgt.gating_mlp(high, low, source_user.squeeze(1), target_item)
        flat_bridge = alpha[:, None] * high + (1-alpha[:, None]) * low
        bridge = flat_bridge.reshape(-1, self.emb_dim, self.emb_dim)
        target_user = torch.bmm(source_user, bridge).squeeze(1)
        rating = torch.sum(target_user * target_item, dim=1)
        return rating, alpha

    def forward(self, x, stage):
        if stage in {"train_meta", "test_meta"}:
            return self.forward_with_gate(x)[0]
        return super().forward(x, stage)
