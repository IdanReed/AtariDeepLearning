# mgdt_tokenizer.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn

from encoder_patch import AtariPatchEncoder


@dataclass
class TokenizerOutput:
    tokens: Tensor          # (B, L, d_model)
    rtg_ids: Tensor         # (B, T)
    game_ids: Tensor        # (B, T)
    action_ids: Tensor      # (B, T)
    B: int
    T: int
    num_patches: int
    tokens_per_step: int    # 3 + num_patches (RTG + GAME + PATCHES + ACTION)


class MGDTTokenizer(nn.Module):
    """
    Multi-Game DT tokenizer (simplified version of the paper’s scheme):

      For each time step t we build:
        [ RTG_t , GAME_t , PATCH_t^1 … PATCH_t^M , ACTION_t ]

      Where:
        - RTG_t is a quantized return-to-go token
        - GAME_t is a game id token
        - PATCH_t^k are spatial image patch tokens from AtariPatchEncoder
        - ACTION_t is a discrete action token

      Final output is a 1D token sequence per batch:
        shape (B, T * (M + 3), d_model)
    """

    def __init__(
        self,
        patch_encoder: AtariPatchEncoder,
        n_actions: int,
        n_games: int = 1,
        rtg_min: int = -20,
        rtg_max: int = 100,
    ) -> None:
        super().__init__()

        self.patch_encoder = patch_encoder
        self.d_model = patch_encoder.d_model

        self.n_actions = n_actions
        self.n_games = n_games

        # RTG quantization range [rtg_min, rtg_max] with bin size 1
        self.rtg_min = rtg_min
        self.rtg_max = rtg_max
        self.n_rtg_bins = (rtg_max - rtg_min) + 1

        # Embeddings
        self.rtg_embed = nn.Embedding(self.n_rtg_bins, self.d_model)
        self.game_embed = nn.Embedding(self.n_games, self.d_model)
        self.action_embed = nn.Embedding(self.n_actions, self.d_model)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _quantize_rtg(self, rtg: Tensor) -> Tensor:
        """
        Quantize continuous RTG values into integer bins [0, n_rtg_bins-1].

        rtg: (B, T) float
        """
        rtg_clipped = rtg.clamp(self.rtg_min, self.rtg_max)
        rtg_ids = torch.round(rtg_clipped - self.rtg_min).long()
        return rtg_ids

    def _broadcast_game_ids(self, game_ids: Tensor, T: int) -> Tensor:
        """
        Accepts either:
          - (B,) game_ids -> broadcast to (B, T)
          - (B, T) game_ids -> returned as-is
        """
        if game_ids.dim() == 1:
            return game_ids.unsqueeze(1).expand(-1, T)
        elif game_ids.dim() == 2 and game_ids.size(1) == T:
            return game_ids
        else:
            raise ValueError(
                f"game_ids must be shape (B,) or (B,T), got {tuple(game_ids.shape)}"
            )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        frames: Tensor,        # (B, T, C=1, 84, 84)
        actions: Tensor,       # (B, T) int
        rtg: Tensor,           # (B, T) float
        game_ids: Optional[Tensor] = None,
    ) -> TokenizerOutput:
        """
        Returns:
          TokenizerOutput where:
            tokens: (B, L, d_model), L = T * (M + 3)
        """
        if frames.dim() != 5:
            raise ValueError(f"frames must be (B,T,C,H,W), got {frames.shape}")
        if actions.dim() != 2:
            raise ValueError(f"actions must be (B,T), got {actions.shape}")
        if rtg.dim() != 2:
            raise ValueError(f"rtg must be (B,T), got {rtg.shape}")

        B, T, C, H, W = frames.shape

        # Default game ids = 0 (single-game setup)
        if game_ids is None:
            game_ids = torch.zeros(B, dtype=torch.long, device=frames.device)

        game_ids_BT = self._broadcast_game_ids(game_ids, T)  # (B,T)

        # 1) Encode image patches
        #    frames -> (B, T, M, d_model)
        patch_tokens = self.patch_encoder(frames)
        if patch_tokens.dim() != 4:
            raise RuntimeError(
                f"patch_encoder must return (B,T,M,d_model), got {patch_tokens.shape}"
            )
        _, _, M, D = patch_tokens.shape
        assert D == self.d_model

        # 2) Discrete token ids
        action_ids = actions.long()              # (B,T)
        rtg_ids = self._quantize_rtg(rtg)        # (B,T)

        # 3) Embeddings
        rtg_tok = self.rtg_embed(rtg_ids)        # (B,T,d_model)
        game_tok = self.game_embed(game_ids_BT)  # (B,T,d_model)
        action_tok = self.action_embed(action_ids)  # (B,T,d_model)

        # 4) Per-step token block: [RTG, GAME, PATCH_1..M, ACTION]
        rtg_tok = rtg_tok.unsqueeze(2)       # (B,T,1,D)
        game_tok = game_tok.unsqueeze(2)     # (B,T,1,D)
        action_tok = action_tok.unsqueeze(2) # (B,T,1,D)

        # concat along "within-step" axis
        step_tokens = torch.cat(
            [rtg_tok, game_tok, patch_tokens, action_tok], dim=2
        )  # (B,T, M+3, D)

        tokens_per_step = M + 3

        # 5) Flatten over time: (B, T*(M+3), D)
        B2, T2, S, D2 = step_tokens.shape
        assert B2 == B and T2 == T and S == tokens_per_step and D2 == self.d_model

        tokens = step_tokens.view(B, T * tokens_per_step, self.d_model)

        return TokenizerOutput(
            tokens=tokens,
            rtg_ids=rtg_ids,
            game_ids=game_ids_BT,
            action_ids=action_ids,
            B=B,
            T=T,
            num_patches=M,
            tokens_per_step=tokens_per_step,
        )
