import torch
import torch.nn as nn

INPUT_MODES = ("full", "scalar_only", "embedding_only")
MODE_TAGS = {"full": "", "scalar_only": "_scalaronly", "embedding_only": "_embeddingonly"}


class RouterNet(nn.Module):
    """
    Two-head router for learn-to-defer escalation.

    - scalar head: Stage-I RM scores, margin, and token counts
    - embedding head: concatenated chosen/rejected response embeddings
    """

    def __init__(
        self,
        emb_in_dim: int = 32,
        embed_out_dim: int = 5,
        explicit_in_dim: int = 4,
        head_scalar_dim: int = 4,
        head_embedding_dim: int = 5,
        combine_hidden: int = 32,
        stage1_index: int = 0,
        input_mode: str = "full",
    ):
        super().__init__()
        self.stage1_index = stage1_index
        assert input_mode in INPUT_MODES, f"input_mode must be one of {INPUT_MODES}"
        self.input_mode = input_mode

        self.scalar_head = nn.Sequential(
            nn.Linear(explicit_in_dim, 16),
            nn.ReLU(),
            nn.Linear(16, head_scalar_dim),
            nn.ReLU(),
        )

        self.embedding_head = nn.Sequential(
            nn.Linear(emb_in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, head_embedding_dim),
            nn.ReLU(),
        )

        if input_mode == "full":
            combine_in_dim = head_scalar_dim + head_embedding_dim
        elif input_mode == "embedding_only":
            combine_in_dim = head_embedding_dim
        else:
            combine_in_dim = head_scalar_dim

        self.combination = nn.Sequential(
            nn.Linear(combine_in_dim, combine_hidden),
            nn.ReLU(),
            nn.Linear(combine_hidden, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(
        self,
        chosen_scores: torch.Tensor,
        rejected_scores: torch.Tensor,
        tokens_chosen: torch.Tensor,
        tokens_rejected: torch.Tensor,
        emb_cat: torch.Tensor,
    ):
        h_embed = self.embedding_head(emb_cat)

        stage1_chosen = chosen_scores[:, self.stage1_index].unsqueeze(-1)
        stage1_rejected = rejected_scores[:, self.stage1_index].unsqueeze(-1)
        stage1_margin_abs = torch.abs(stage1_chosen - stage1_rejected)
        stage1_token_diff_abs = torch.abs(
            tokens_chosen[:, self.stage1_index].unsqueeze(-1)
            - tokens_rejected[:, self.stage1_index].unsqueeze(-1)
        )
        explicit_features = torch.cat(
            [stage1_chosen, stage1_rejected, stage1_margin_abs, stage1_token_diff_abs],
            dim=-1,
        )
        h_scalar = self.scalar_head(explicit_features)

        if self.input_mode == "embedding_only":
            combined = h_embed
        elif self.input_mode == "scalar_only":
            combined = h_scalar
        else:
            combined = torch.cat([h_scalar, h_embed], dim=-1)

        logit = self.combination(combined).squeeze(-1)
        prob = torch.sigmoid(logit)
        return logit, prob
