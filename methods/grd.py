"""GRD: Gated Reversal Decoding.

GRD is stateful across a generation: it first decides the trusted branch at the
first context/prior top-1 conflict, then uses the pairwise reversal point tau*
on prior-trusted conflict tokens.  Agreement tokens always emit the shared
context/prior top-1 token.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class GRDDecoding:
    """Stateful GRD controller for one generated sequence."""

    name = "grd"

    def __init__(self, grd_lambda: float = 0.75, gate_mode: str = "full"):
        valid_modes = {"full", "conf_only", "ent_only"}
        if gate_mode not in valid_modes:
            raise ValueError(f"Unsupported GRD gate_mode: {gate_mode!r}")
        self.grd_lambda = grd_lambda
        self.gate_mode = gate_mode
        self.trusted = "ctx"
        self.gate_decided = False

    def reset(self) -> None:
        self.trusted = "ctx"
        self.gate_decided = False

    def select_next_token(self, logits_ctx: torch.Tensor, logits_prior: torch.Tensor) -> int:
        """Return the next token id from one ctx/prior logits pair."""
        lp_ctx = F.log_softmax(logits_ctx.float(), dim=-1)
        lp_prior = F.log_softmax(logits_prior.float(), dim=-1)
        a_ctx = int(torch.argmax(lp_ctx).item())
        a_prior = int(torch.argmax(lp_prior).item())

        if a_ctx != a_prior and not self.gate_decided:
            p_prior_top = float(lp_prior[a_prior].exp().item())
            h_prior = float(-(lp_prior.exp() * lp_prior).sum().item())
            h_ctx = float(-(lp_ctx.exp() * lp_ctx).sum().item())
            conf_signal = p_prior_top > 0.5
            ent_signal = h_prior < h_ctx
            if self.gate_mode == "conf_only":
                route_to_prior = conf_signal
            elif self.gate_mode == "ent_only":
                route_to_prior = ent_signal
            else:
                route_to_prior = conf_signal and ent_signal
            self.trusted = "prior" if route_to_prior else "ctx"
            self.gate_decided = True

        if a_ctx == a_prior:
            return a_ctx
        if self.trusted == "ctx":
            return a_ctx

        m_c = float((lp_ctx[a_ctx] - lp_ctx[a_prior]).item())
        m_p = float((lp_prior[a_prior] - lp_prior[a_ctx]).item())
        denominator = m_c + m_p
        tau_star = m_p / denominator if abs(denominator) > 1e-9 else 1.0
        tau = (1.0 - self.grd_lambda) * tau_star
        mixed = tau * lp_ctx + (1.0 - tau) * lp_prior
        return int(torch.argmax(mixed).item())
