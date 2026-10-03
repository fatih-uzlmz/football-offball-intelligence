"""Value model V2: per-frame player attention + temporal encoder.

Input:  [B, T=75, N=23, F=8]  (3s window; 22 players + ball as tokens)
Output: P(shot within 5s), P(ball in box within 5s), P(line break within 5s)

Per frame, a Transformer encoder runs self-attention over the 23 player
tokens (with 23 nodes this is exactly a GAT on a complete graph), then a
learned query pools the tokens into one frame embedding. An LSTM reads the
75 frame embeddings; an MLP head emits 3 logits.

The per-frame attention weights are returned on request: they show which
players the model looked at, making the value estimate inspectable.
"""
import torch
import torch.nn as nn

TARGETS = ["shot", "box_entry", "line_break"]


class PlayerAttention(nn.Module):
    """GAT-style self-attention over players within one frame + pooling.

    One multi-head self-attention layer (complete graph over the 23 player
    tokens) with residual + norm — deliberately no transformer FFN: on CPU
    the FFN dominates cost and adds little for 23 tokens. A learned query
    then pools the tokens into one frame embedding; the pooling weights
    show which players the model looked at.
    """

    def __init__(self, d_feat=8, d_model=64, n_heads=4, dropout=0.2):
        super().__init__()
        self.embed = nn.Linear(d_feat, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.query = nn.Parameter(torch.randn(d_model) * 0.1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, return_attn=False):
        # x: [B*T, N, F] -> frame embeddings [B*T, D]
        h = self.embed(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        h = self.norm(h + self.drop(a))
        w = torch.softmax(h @ self.query, dim=1)  # [B*T, N] player weights
        pooled = (w.unsqueeze(-1) * h).sum(1)     # [B*T, D]
        if return_attn:
            return pooled, w
        return pooled


class GraphValue(nn.Module):
    def __init__(self, n_feat=8, d_model=64, n_heads=4,
                 t_hidden=64, t_layers=2, n_targets=3, dropout=0.2):
        super().__init__()
        self.player_attn = PlayerAttention(n_feat, d_model, n_heads, dropout)
        self.lstm = nn.LSTM(d_model, t_hidden, t_layers,
                            batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(t_hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_targets),
        )

    def forward(self, x, return_attn=False):
        # x: [B, T, N, F]
        B, T, N, F = x.shape
        if return_attn:
            pooled, w = self.player_attn(x.reshape(B * T, N, F),
                                         return_attn=True)
            w = w.reshape(B, T, N)
        else:
            pooled = self.player_attn(x.reshape(B * T, N, F))
            w = None
        frames = pooled.reshape(B, T, -1)        # [B, T, D]
        _, (h, _) = self.lstm(frames)
        logits = self.head(h[-1])                # [B, n_targets]
        if return_attn:
            return logits, w
        return logits
