# mgdt_model.py

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoders import ObsEncoder


class MGDTModel(nn.Module):
    """
    Notes:
    - Our data set is prov
    Multi-Game Decision Transformer-style model with a pluggable ObsEncoder.

    For each timestep t in [0, T-1], we create tokens:
      [ o_t^1, ..., o_t^M, R_t, a_t, r_t ]

    where:
      - o_t^m   = observation patch tokens (M = num_tokens_per_frame)
      - R_t     = return-to-go bin
      - a_t     = action (discrete)
      - r_t     = reward bin

    The sequence over a window of T timesteps is:
      [ o_0^1 ... o_0^M R_0 a_0 r_0  o_1^1 ... R_1 a_1 r_1  ... ]

    Masking:
      - Standard causal mask over timesteps/tokens, so tokens cannot see the future.
      - BUT all patch tokens from the same timestep can attend to each other
        bidirectionally (full attention within that patch block).
    """

    TYPE_OBS = 0
    TYPE_RETURN = 1
    TYPE_ACTION = 2
    TYPE_REWARD = 3

    def __init__(
        self,
        obs_encoder: ObsEncoder,
        n_actions: int,
        n_return_bins: int,
        n_reward_bins: int = 3,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        dim_feedforward: int = 2048,
        max_timestep_window_size: int = 32,
        dropout: float = 0.1,
        loss_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ):
        super().__init__()

        self.obs_encoder = obs_encoder
        self.d_model = d_model
        self.n_actions = n_actions
        self.n_return_bins = n_return_bins
        self.n_reward_bins = n_reward_bins
        self.max_timestep_window_size = max_timestep_window_size
        self.loss_weights = loss_weights

        self.L_obs = obs_encoder.num_tokens_per_frame
        self.tokens_per_step = self.L_obs + 3  # obs + (return, action, reward)
        self.max_seq_len = self.tokens_per_step * max_timestep_window_size

        # token embeddings
        self.return_embed = nn.Embedding(n_return_bins, d_model)
        self.action_embed = nn.Embedding(n_actions, d_model)
        self.reward_embed = nn.Embedding(n_reward_bins, d_model)

        # positional + type embeddings
        self.pos_embed = nn.Embedding(self.max_seq_len, d_model)
        self.type_embed = nn.Embedding(4, d_model)  # obs/return/action/reward

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,  # (batch_size, sequence_length, d_model)
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False, # idk there was a warning about this
        )

        # heads
        self.return_head = nn.Linear(d_model, n_return_bins)
        self.action_head = nn.Linear(d_model, n_actions)
        self.reward_head = nn.Linear(d_model, n_reward_bins)

        # cache for attention masks keyed by timestep_window_size
        self._mask_cache: Dict[int, torch.Tensor] = {}

    # ---- token layout helpers ----

    def _build_seq_tokens(
        self,
        frames: torch.Tensor,      # (batch_size, timesteps, channels, height, width)
        rtg_bins: torch.Tensor,    # (batch_size, timesteps)
        actions: torch.Tensor,     # (batch_size, timesteps)
        reward_bins: torch.Tensor, # (batch_size, timesteps)
    ) -> torch.Tensor:
        """
        Build token embeddings (with pos/type added) of shape (batch_size, sequence_length, d_model),
        where sequence_length = timesteps * (L_obs + 3).
        """
        B, T, C, H, W = frames.shape

        # obs tokens from encoder: (batch_size, timesteps, L_obs, d_model)
        obs_tokens = self.obs_encoder(frames)

        # embed scalar tokens
        R_emb = self.return_embed(rtg_bins)      # (batch_size, timesteps, d_model)
        A_emb = self.action_embed(actions)       # (batch_size, timesteps, d_model)
        r_emb = self.reward_embed(reward_bins)   # (batch_size, timesteps, d_model)

        R_emb = R_emb.unsqueeze(2)  # (batch_size, timesteps, 1, d_model)
        A_emb = A_emb.unsqueeze(2)  # (batch_size, timesteps, 1, d_model)
        r_emb = r_emb.unsqueeze(2)  # (batch_size, timesteps, 1, d_model)

        # concat per timestep: (batch_size, timesteps, L_obs+3, d_model)
        step_tokens = torch.cat(
            [obs_tokens, R_emb, A_emb, r_emb],
            dim=2,
        )
        B, T, S, D = step_tokens.shape  # tokens_per_step = L_obs + 3

        # flatten over timesteps: (batch_size, sequence_length, d_model)
        seq_tokens = step_tokens.view(B, T * S, D)

        device = frames.device
        L = T * S

        # type ids per position (shared across batch)
        pos = torch.arange(L, device=device)
        step = pos // S
        offset = pos % S

        type_ids = torch.empty(L, dtype=torch.long, device=device)
        type_ids[offset < self.L_obs] = self.TYPE_OBS
        type_ids[offset == self.L_obs] = self.TYPE_RETURN
        type_ids[offset == self.L_obs + 1] = self.TYPE_ACTION
        type_ids[offset == self.L_obs + 2] = self.TYPE_REWARD

        type_ids = type_ids.unsqueeze(0).expand(B, -1)  # (batch_size, sequence_length)
        pos_ids = pos.unsqueeze(0).expand(B, -1)        # (batch_size, sequence_length)

        seq_tokens = seq_tokens + self.pos_embed(pos_ids) + self.type_embed(type_ids)
        return seq_tokens

    def _build_attention_mask(self, timestep_window_size: int, device: torch.device) -> torch.Tensor:
        """
        Build (sequence_length, sequence_length) attention mask with:
          - Causal structure overall (no attending to future tokens).
          - BUT patch tokens from the same timestep may attend each other
            bidirectionally.

        Mask format is additive:
          - 0.0 for allowed attention
          - -inf for disallowed attention
        """
        if timestep_window_size in self._mask_cache:
            return self._mask_cache[timestep_window_size].to(device)

        S = self.tokens_per_step
        L = timestep_window_size * S

        pos = torch.arange(L, device=device)
        step = pos // S          # timestep index per position
        offset = pos % S         # position within timestep
        is_obs = offset < self.L_obs

        # base: causal upper-triangular mask
        mask = torch.triu(
            torch.full((L, L), float("-inf"), device=device),
            diagonal=1,
        )

        # allow full attention between obs tokens within the same timestep
        same_step = step.unsqueeze(0) == step.unsqueeze(1)  # (sequence_length, sequence_length)
        obs_pair = (
            is_obs.unsqueeze(0)
            & is_obs.unsqueeze(1)
            & same_step
        )  # (sequence_length, sequence_length) bool
        mask = torch.where(obs_pair, torch.zeros_like(mask), mask)

        # cache on CPU
        self._mask_cache[timestep_window_size] = mask.detach().cpu()
        return mask

    def _indices_per_type(
        self,
        timestep_window_size: int,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """
        Return boolean masks over sequence positions for return/action/reward tokens.
        """
        S = self.tokens_per_step
        L = timestep_window_size * S
        pos = torch.arange(L, device=device)
        offset = pos % S

        is_return = offset == self.L_obs
        is_action = offset == self.L_obs + 1
        is_reward = offset == self.L_obs + 2

        return {
            "is_return": is_return,  # (sequence_length,)
            "is_action": is_action,
            "is_reward": is_reward,
        }

    # ---- forward + loss ----

    def forward(
        self,
        frames: torch.Tensor,      # (batch_size, timesteps, channels, height, width)
        rtg_bins: torch.Tensor,    # (batch_size, timesteps)
        actions: torch.Tensor,     # (batch_size, timesteps)
        reward_bins: torch.Tensor, # (batch_size, timesteps)
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning logits for return, action, and reward heads.

        All three are defined for every timestep in the window.
        """
        B, T, C, H, W = frames.shape
        seq_tokens = self._build_seq_tokens(frames, rtg_bins, actions, reward_bins)
        device = frames.device

        attn_mask = self._build_attention_mask(T, device=device)  # (sequence_length, sequence_length)
        # TransformerEncoder expects (batch_size, sequence_length, d_model) with batch_first=True
        h = self.transformer(seq_tokens, mask=attn_mask)          # (batch_size, sequence_length, d_model)

        idx = self._indices_per_type(T, device=device)
        # boolean masks along sequence dimension
        is_return = idx["is_return"]
        is_action = idx["is_action"]
        is_reward = idx["is_reward"]

        # pick positions for each token type: (batch_size, timesteps, d_model)
        return_h = h[:, is_return, :]
        action_h = h[:, is_action, :]
        reward_h = h[:, is_reward, :]

        return_logits = self.return_head(return_h)   # (batch_size, timesteps, n_return_bins)
        action_logits = self.action_head(action_h)   # (batch_size, timesteps, n_actions)
        reward_logits = self.reward_head(reward_h)   # (batch_size, timesteps, n_reward_bins)

        return {
            "return_logits": return_logits,
            "action_logits": action_logits,
            "reward_logits": reward_logits,
        }

    def compute_loss(
        self,
        frames: torch.Tensor,
        rtg_bins: torch.Tensor,
        actions: torch.Tensor,
        reward_bins: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss and per-head losses.

        Targets are:
          - rtg_bins:      (batch_size, timesteps) in [0, n_return_bins)
          - actions:       (batch_size, timesteps) in [0, n_actions)
          - reward_bins:   (batch_size, timesteps) in [0, n_reward_bins)
        """
        out = self.forward(frames, rtg_bins, actions, reward_bins)
        B, T = actions.shape

        loss_R = F.cross_entropy(
            out["return_logits"].view(B * T, -1),
            rtg_bins.view(-1),
        )
        loss_A = F.cross_entropy(
            out["action_logits"].view(B * T, -1),
            actions.view(-1),
        )
        loss_r = F.cross_entropy(
            out["reward_logits"].view(B * T, -1),
            reward_bins.view(-1),
        )

        w_R, w_A, w_r = self.loss_weights
        total = w_R * loss_R + w_A * loss_A + w_r * loss_r

        stats = {
            "loss": float(total.item()),
            "loss_return": float(loss_R.item()),
            "loss_action": float(loss_A.item()),
            "loss_reward": float(loss_r.item()),
        }
        return total, stats
