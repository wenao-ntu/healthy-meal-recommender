"""A deterministic positive-frequency baseline compatible with RecBole evaluation."""

from __future__ import annotations

import torch
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import InputType, ModelType


class MostPop(GeneralRecommender):
    """Score every item by its exact positive frequency in the training dataset."""

    input_type = InputType.POINTWISE
    type = ModelType.TRADITIONAL

    def __init__(self, config, dataset) -> None:
        super().__init__(config, dataset)
        item_ids = dataset.inter_feat[self.ITEM_ID]
        item_counts = torch.bincount(item_ids, minlength=self.n_items).to(torch.float64)
        self.register_buffer("item_cnt", item_counts)
        self.register_buffer("max_cnt", item_counts.max().clamp_min(1))
        internal_ids = torch.arange(self.n_items, dtype=torch.float64)
        tie_break = (self.n_items - internal_ids) / self.n_items * 1e-9
        self.register_buffer("ranking_score", item_counts / self.max_cnt + tie_break)
        self.optimizer_placeholder = torch.nn.Parameter(torch.zeros(1))

    def calculate_loss(self, interaction) -> torch.Tensor:
        """Return a no-op loss if a generic framework training hook calls this model."""
        return self.optimizer_placeholder * 0

    def predict(self, interaction) -> torch.Tensor:
        """Return normalized training frequency for requested items."""
        item_ids = interaction[self.ITEM_ID]
        return self.ranking_score[item_ids]

    def full_sort_predict(self, interaction) -> torch.Tensor:
        """Return the same catalog score vector for each user."""
        user_count = interaction[self.USER_ID].shape[0]
        return self.ranking_score.repeat(user_count)
