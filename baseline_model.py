# mgdt_model.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MGDTOutput:
    logits: Tensor          # (B, T, n_actions) - action logits at each timestep
    hidden: Tensor          # (B, L, d_model)   - full sequence hidden states
    action_positions: Tensor  # (B, T) indices into sequence dimension


class MultiGameDecisionTransformer(nn.Module):
    """
    Causal transformer over flattened token sequences produced by MGDTTokenizer.

    Input:
      tokens: (B, L, d_model)
      tokens_per_step: int   (e.g. 39 = 1 RTG + 1 GAME + 36 PATCH + 1 ACTION)
      T: int                 number of timesteps per sequence

    We assume the last token in each step is the action token.
    """

    def __init__(
        self,
        d_model: int,
        n_actions: int,
        n_layers: int = 4,
        n_heads: int = 4,
        dim_feedforward: int = 4 * 128,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.n_actions = n_actions
        self.max_seq_len = max_seq_len

        # Positional embedding over flattened sequence positions [0..L-1]
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,  # (B, L, D)
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        # Predict discrete action logits from hidden states at action positions
        self.action_head = nn.Linear(d_model, n_actions)

    # ------------------------------------------------------------------
    # Causal mask helper
    # ------------------------------------------------------------------
    def _causal_mask(self, L: int, device: torch.device) -> Tensor:
        """
        Create (L, L) mask with -inf above diagonal, 0 elsewhere.
        This prevents attending to future tokens.
        """
        mask = torch.full((L, L), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        tokens: Tensor,        # (B, L, d_model)
        tokens_per_step: int,  # e.g. 39
        T: int,                # timesteps
    ) -> MGDTOutput:
        B, L, D = tokens.shape
        if D != self.d_model:
            raise ValueError(f"d_model mismatch: expected {self.d_model}, got {D}")

        if L != tokens_per_step * T:
            raise ValueError(
                f"L must equal tokens_per_step * T, got L={L}, tokens_per_step={tokens_per_step}, T={T}"
            )

        if L > self.max_seq_len:
            raise ValueError(
                f"Sequence length L={L} exceeds max_seq_len={self.max_seq_len}"
            )

        device = tokens.device

        # 1) Add positional embeddings
        pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, L)  # (B,L)
        x = tokens + self.pos_embed(pos_ids)  # (B,L,D)

        # 2) Causal mask
        attn_mask = self._causal_mask(L, device)  # (L,L)

        # 3) Transformer encoder
        hidden = self.transformer(x, mask=attn_mask)  # (B,L,D)

        # 4) Select action token positions
        S = tokens_per_step
        # action index within each step = S - 1
        # positions: [S-1, 2S-1, 3S-1, ..., T*S-1]
        action_pos_1d = torch.arange(T, device=device) * S + (S - 1)  # (T,)
        action_pos = action_pos_1d.unsqueeze(0).expand(B, T)          # (B,T)

        # Gather hidden states at these positions
        # hidden: (B,L,D), index: (B,T,1) -> gathered: (B,T,D)
        gather_index = action_pos.unsqueeze(-1).expand(-1, -1, D)     # (B,T,D)
        hidden_actions = torch.gather(hidden, dim=1, index=gather_index)  # (B,T,D)

        # 5) Action logits
        logits = self.action_head(hidden_actions)  # (B,T,n_actions)

        return MGDTOutput(
            logits=logits,
            hidden=hidden,
            action_positions=action_pos,
        )
