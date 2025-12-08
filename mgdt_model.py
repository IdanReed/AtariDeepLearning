from __future__ import annotations

from typing import Dict, List, Tuple
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoders import ObsEncoder


class Freezeable(Enum):
    """
    These names must match the attribute names of the model
    """
    Transformer = "transformer"
    ObsEncoder = "obs_encoder"
    # I don't think these are really needed but adding them for completeness
    ReturnEmbed = "return_embed"
    ActionEmbed = "action_embed"
    RewardEmbed = "reward_embed"
    PosEmbed = "pos_embed"
    TypeEmbed = "type_embed"
    ReturnHead = "return_head"
    ActionHead = "action_head"
    RewardHead = "reward_head"

class MGDTModel(nn.Module):
    TYPE_OBS = 0
    TYPE_RETURN = 1
    TYPE_ACTION = 2
    TYPE_REWARD = 3

    def __init__(
        self,
        obs_encoder: ObsEncoder,
        n_actions: int,
        n_return_bins: int,
        n_games: int,
        n_reward_bins: int = 3,
        emb_size: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        dim_feedforward: int = 2048,
        max_timestep_window_size: int = 32,
        dropout: float = 0.1,
        loss_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ):
        super().__init__()

        self.obs_encoder = obs_encoder
        self.d_model = emb_size
        self.n_actions = n_actions
        self.n_return_bins = n_return_bins
        self.n_games = n_games
        self.n_reward_bins = n_reward_bins
        self.max_timestep_window_size = max_timestep_window_size
        self.loss_weights = loss_weights

        self.num_tokens_per_frame = obs_encoder.num_tokens_per_frame
        self.tokens_per_timestep = self.num_tokens_per_frame + 3  # obs + (return, action, reward)
        self.max_seq_len = self.tokens_per_timestep * max_timestep_window_size

        # token embeddings for scalar tokens
        self.return_embed = nn.Embedding(n_return_bins, emb_size)
        self.action_embed = nn.Embedding(n_actions, emb_size)
        self.reward_embed = nn.Embedding(n_reward_bins, emb_size)
        self.game_embed = nn.Embedding(n_games, emb_size)



        # positional + type embeddings
        self.pos_embed = nn.Embedding(self.max_seq_len, emb_size)
        self.type_embed = nn.Embedding(4, emb_size)  # obs/return/action/reward

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,  # (batch_size, tokens_per_seq, emb_size)
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )

        self.return_head = nn.Linear(emb_size, n_return_bins)
        self.action_head = nn.Linear(emb_size, n_actions)
        self.reward_head = nn.Linear(emb_size, n_reward_bins)

        self._mask_cache: Dict[int, torch.Tensor] = {}

    def _build_seq_tokens(
        self,
        frames: torch.Tensor,      # (B, T, C, H, W)
        rtg_bins: torch.Tensor,    # (B, T)
        actions: torch.Tensor,     # (B, T)
        reward_bins: torch.Tensor, # (B, T)
        game_ids: torch.Tensor,    # (B, )
    ) -> torch.Tensor:
        """
        Build token embeddings (with pos/type added) of shape (B, L, D),
        where L = timesteps * (L_obs + 3).
        """
        batch_size, n_timesteps, n_channels, img_height, img_width = frames.shape

        # obs tokens from encoder
        obs_tokens = self.obs_encoder(frames)                           # (B, n_timesteps, num_tokens_per_frame, D)

        # embed scalar tokens
        return_emb = self.return_embed(rtg_bins)                        # (B, n_timesteps, D)
        action_emb = self.action_embed(actions)                         # (B, n_timesteps, D)
        reward_emb = self.reward_embed(reward_bins)                     # (B, n_timesteps, D)

        return_emb = return_emb.unsqueeze(2)                            # (B, n_timesteps, 1, D)
        action_emb = action_emb.unsqueeze(2)                            # (B, n_timesteps, 1, D)
        reward_emb = reward_emb.unsqueeze(2)                            # (B, n_timesteps, 1, D)

        # concat per timestep
        step_tokens = torch.cat(                                        # (B, n_timesteps, num_tokens_per_frame+3, D)
            [obs_tokens, return_emb, action_emb, reward_emb],
            dim=2,
        )
        batch_size, n_timesteps, tokens_per_timestep, emb_size = step_tokens.shape

        # flatten
        seq_tokens = step_tokens.view(batch_size, n_timesteps * tokens_per_timestep, emb_size)  # (B, seq_len, emb_size)     where seq_len = (n_timesteps * tokens_per_timestep)

        device = frames.device
        L = n_timesteps * tokens_per_timestep

        # positions
        pos = torch.arange(L, device=device)
        pos_ids = pos.unsqueeze(0).expand(batch_size, -1)        # (B, seq_len)

        # types
        offset = pos % tokens_per_timestep
        type_ids = torch.empty(L, dtype=torch.long, device=device)
        type_ids[offset < self.num_tokens_per_frame] = self.TYPE_OBS
        type_ids[offset == self.num_tokens_per_frame] = self.TYPE_RETURN
        type_ids[offset == self.num_tokens_per_frame + 1] = self.TYPE_ACTION
        type_ids[offset == self.num_tokens_per_frame + 2] = self.TYPE_REWARD
        type_ids = type_ids.unsqueeze(0).expand(batch_size, -1)  # (B, seq_len)
        
        seq_tokens = seq_tokens + self.pos_embed(pos_ids) + self.type_embed(type_ids)

        game_emb = self.game_embed(game_ids)
        game_emb = game_emb.unsqueeze(1)
        seq_tokens = seq_tokens + game_emb

        return seq_tokens

    def _build_attention_mask(self, n_timesteps: int, device: torch.device) -> torch.Tensor:
        """
        This is confusing as hell, but we've got multiple timesteps each with obs tokens and R/a/r tokens all in one sequence.

        Generally a transformer attention mask is always causal, meaning that a token can only attend to previous tokens.
        
        However, obs tokens are not sequential so we need to allow them to attend to each other bidirectionally (that's what the paper does).
        """
        if n_timesteps in self._mask_cache:
            return self._mask_cache[n_timesteps].to(device)

        tokens_per_timestep = self.tokens_per_timestep
        seq_len = n_timesteps * tokens_per_timestep

        pos = torch.arange(seq_len, device=device)
        step = pos // tokens_per_timestep          # timestep index per position
        offset = pos % tokens_per_timestep         # position within timestep
        is_obs = offset < self.num_tokens_per_frame

        # base: causal upper-triangular mask
        mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1,
        )

        # allow full attention between obs tokens within the same timestep
        same_step = step.unsqueeze(0) == step.unsqueeze(1)  # (L, L)
        obs_pair = (
            is_obs.unsqueeze(0)
            & is_obs.unsqueeze(1)
            & same_step
        )
        mask = torch.where(obs_pair, torch.zeros_like(mask), mask)

        # cache
        self._mask_cache[n_timesteps] = mask.detach().cpu()
        return mask

    def _indices_per_type(
        self,
        n_timesteps: int,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        tokens_per_timestep = self.tokens_per_timestep
        seq_len = n_timesteps * tokens_per_timestep
        pos = torch.arange(seq_len, device=device)
        offset = pos % tokens_per_timestep

        # label positions
        is_return = offset == self.num_tokens_per_frame
        is_action = offset == self.num_tokens_per_frame + 1
        is_reward = offset == self.num_tokens_per_frame + 2

        # source positions (previous tokens):
        #   R_t  : last obs token at this timestep
        #   a_t  : R_t token
        #   r_t  : a_t token
        src_for_return = offset == (self.num_tokens_per_frame - 1)
        src_for_action = is_return
        src_for_reward = is_action

        return {
            "is_return": is_return,
            "is_action": is_action,
            "is_reward": is_reward,
            "src_for_return": src_for_return,
            "src_for_action": src_for_action,
            "src_for_reward": src_for_reward,
        }

    def forward(
        self,
        frames: torch.Tensor,      # (B, T, C, H, W)
        rtg_bins: torch.Tensor,    # (B, T)
        actions: torch.Tensor,     # (B, T)
        reward_bins: torch.Tensor, # (B, T)
        game_ids: torch.Tensor,    # (B, )
    ) -> Dict[str, torch.Tensor]:
        # Get tokens
        batch_size, n_timesteps, n_channels, img_height, img_width = frames.shape
        seq_tokens = self._build_seq_tokens(frames, rtg_bins, actions, reward_bins, game_ids)
        device = frames.device

        # Attention mask
        attn_mask = self._build_attention_mask(n_timesteps, device=device)  # (L, L)

        # TRANSFORMER
        hidden_states = self.transformer(seq_tokens, mask=attn_mask)          # (B, seq_len, emb_size)

        # These dimensions are messing with me, but we've got a big block of hidden state
        # Same dimensions as transformer input but now everything is attented to each other

        # This gives us the TOKEN INDEX for the token preceding what we're predicting
        # For each time step, within each batch
        idx = self._indices_per_type(n_timesteps, device=device)
        src_ret = idx["src_for_return"]
        src_act = idx["src_for_action"]
        src_rew = idx["src_for_reward"]

        # For each timestep within each batch pull out a vector that is emb_size 
        # e.g. (batch_size, n_timesteps, emb_size)
        return_h = hidden_states[:, src_ret, :]
        action_h = hidden_states[:, src_act, :]
        reward_h = hidden_states[:, src_rew, :]

        return_logits = self.return_head(return_h)   # (B, T, n_return_bins)
        action_logits = self.action_head(action_h)   # (B, T, n_actions)
        reward_logits = self.reward_head(reward_h)   # (B, T, n_reward_bins)

        return {
            "return_logits": return_logits,
            "action_logits": action_logits,
            "reward_logits": reward_logits,
        }

    def forward_and_compute_loss(
        self,
        frames: torch.Tensor,
        rtg_bins: torch.Tensor,
        actions: torch.Tensor,
        reward_bins: torch.Tensor,
        game_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        out = self.forward(frames, rtg_bins, actions, reward_bins, game_ids)
        
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
        return out, total, stats

    def freeze(self, components: List[Freezeable]) -> None:
        components_frozen = 0
        for component in components:
            module = getattr(self, component.value)
            module.eval()
            for param in module.parameters():
                param.requires_grad = False
            components_frozen += 1
        return components_frozen

    def unfreeze(self, components: List[Freezeable]) -> None:
        components_unfrozen = 0
        for component in components:
            module = getattr(self, component.value)
            module.train()
            for param in module.parameters():
                param.requires_grad = True
            components_unfrozen += 1
        return components_unfrozen