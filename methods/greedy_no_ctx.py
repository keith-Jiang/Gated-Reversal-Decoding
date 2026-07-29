"""Greedy decoding without context: use prior-only logits."""

import torch
from methods.base import DecodingMethod


class GreedyNoCtxDecoding(DecodingMethod):
    name = "greedy_no_ctx"

    def get_next_token_logits(self, logits_ctx, logits_prior):
        return logits_prior

    def get_tau(self, logits_ctx, logits_prior):
        return 0.0
