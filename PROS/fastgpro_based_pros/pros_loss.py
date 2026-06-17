"""PROS advantage and policy loss functions.

The functions mirror the reference implementation in
``PROS/original_pros/verl/trainer/ppo/core_algos.py`` where possible and add
the FastGRPO target loss used by ``FastGRPO/grpo_speculative.py``.
"""

from __future__ import annotations

from typing import Dict, Iterable, Literal, NamedTuple, Optional


def _torch():
    try:
        import torch

        return torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pros_loss requires torch. Install the FastGRPO requirements before running training."
        ) from exc


class ProsLossStats(NamedTuple):
    loss: "object"
    pg_loss: "object"
    kl_loss: "object"
    approx_kl: "object"
    clip_fraction: "object"
    advantage_mean: "object"
    advantage_abs_mean: "object"


def masked_mean(values, mask, eps: float = 1e-8):
    torch = _torch()
    mask = mask.to(dtype=values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(eps)


def aggregate_loss(loss_mat, loss_mask, mode: str = "token-mean"):
    torch = _torch()
    loss_mask = loss_mask.to(dtype=loss_mat.dtype)
    if mode == "token-mean":
        return masked_mean(loss_mat, loss_mask)
    if mode == "seq-mean-token-sum":
        return torch.sum(loss_mat * loss_mask, dim=-1).mean()
    if mode == "seq-mean-token-mean":
        denom = torch.sum(loss_mask, dim=-1).clamp_min(1.0)
        return (torch.sum(loss_mat * loss_mask, dim=-1) / denom).mean()
    if mode == "seq-mean-token-sum-norm":
        return torch.sum(loss_mat * loss_mask) / max(loss_mask.shape[-1], 1)
    raise ValueError(f"Invalid loss_agg_mode: {mode}")


def gather_token_logps(logits, labels):
    """Return next-token log-probabilities for ``labels[:, 1:]``."""

    torch = _torch()
    logits = logits[:, :-1, :].float()
    shifted_labels = labels[:, 1:].to(logits.device)
    return torch.gather(logits.log_softmax(dim=-1), dim=2, index=shifted_labels.unsqueeze(2)).squeeze(2)


def compute_grpo_advantages(
    rewards,
    response_mask,
    group_ids: Iterable[int],
    *,
    normalize_by_std: bool = True,
    epsilon: float = 1e-6,
):
    """Compute token-level GRPO advantages from scalar response rewards."""

    torch = _torch()
    scores = rewards.to(dtype=torch.float32).clone()
    if scores.ndim != 1:
        scores = scores.view(-1)

    group_ids = list(group_ids)
    if len(group_ids) != scores.shape[0]:
        raise ValueError(f"group_ids length {len(group_ids)} does not match rewards batch {scores.shape[0]}")

    advantages = torch.zeros_like(scores)
    for gid in sorted(set(group_ids)):
        idx = [i for i, group_id in enumerate(group_ids) if group_id == gid]
        idx_t = torch.as_tensor(idx, device=scores.device, dtype=torch.long)
        group_scores = scores.index_select(0, idx_t)
        if group_scores.numel() == 1:
            centered = torch.zeros_like(group_scores)
        else:
            centered = group_scores - group_scores.mean()
            if normalize_by_std:
                centered = centered / group_scores.std(unbiased=True).clamp_min(epsilon)
        advantages.index_copy_(0, idx_t, centered)

    return advantages.unsqueeze(-1) * response_mask.to(advantages.dtype)


def compute_gpg_advantages(
    rewards,
    response_mask,
    group_ids: Iterable[int],
    *,
    f_norm: float = 1.0,
):
    """Compute the GPG advantages present in the reference PROS tree.

    This matches the reference behavior: ``alpha = bsz / count_nonzero(scores)``
    and each response is centered by its group mean, without dividing by the
    computed group standard deviation.
    """

    torch = _torch()
    scores = rewards.to(dtype=torch.float32).clone().view(-1)
    group_ids = list(group_ids)
    nonzero = torch.count_nonzero(scores).clamp(min=1)
    alpha = scores.shape[0] / nonzero
    advantages = torch.zeros_like(scores)

    for gid in sorted(set(group_ids)):
        idx = [i for i, group_id in enumerate(group_ids) if group_id == gid]
        idx_t = torch.as_tensor(idx, device=scores.device, dtype=torch.long)
        group_scores = scores.index_select(0, idx_t)
        group_mean = group_scores.mean() if group_scores.numel() > 1 else torch.zeros((), device=scores.device)
        centered = alpha * (group_scores - group_mean) / f_norm
        advantages.index_copy_(0, idx_t, centered)

    return advantages.unsqueeze(-1) * response_mask.to(advantages.dtype)


def compute_fastgrpo_policy_loss(
    log_prob,
    old_log_prob,
    ref_log_prob,
    advantages,
    response_mask,
    *,
    epsilon: float,
    beta: float,
    loss_agg_mode: str,
):
    """FastGRPO target loss with PROS-provided advantages.

    The KL term is ``exp(ref_logp - logp) - (ref_logp - logp) - 1``, matching
    ``FastGRPO/grpo_speculative.py``.
    """

    torch = _torch()
    ratio = torch.exp(torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0))
    clipped_ratio = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
    pg_objective = torch.minimum(ratio * advantages, clipped_ratio * advantages)

    if ref_log_prob is None or beta == 0:
        kl_penalty = torch.zeros_like(pg_objective)
    else:
        ref_delta = torch.clamp(ref_log_prob - log_prob, min=-20.0, max=20.0)
        kl_penalty = torch.exp(ref_delta) - ref_delta - 1.0

    per_token_loss = -(pg_objective - beta * kl_penalty)
    loss = aggregate_loss(per_token_loss, response_mask, loss_agg_mode)
    clip_fraction = masked_mean((ratio != clipped_ratio).to(log_prob.dtype), response_mask)
    approx_kl = masked_mean(old_log_prob - log_prob, response_mask)
    return ProsLossStats(
        loss=loss,
        pg_loss=aggregate_loss(-pg_objective, response_mask, loss_agg_mode),
        kl_loss=aggregate_loss(kl_penalty, response_mask, loss_agg_mode),
        approx_kl=approx_kl,
        clip_fraction=clip_fraction,
        advantage_mean=masked_mean(advantages, response_mask),
        advantage_abs_mean=masked_mean(advantages.abs(), response_mask),
    )


def compute_clipped_grpo_policy_loss(
    log_prob,
    old_log_prob,
    advantages,
    response_mask,
    *,
    epsilon: float,
    loss_agg_mode: str,
):
    """Reference verl-style clipped policy loss used by PROS when GRPO is selected."""

    torch = _torch()
    negative_approx_kl = torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    clipped_ratio = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
    pg_losses = torch.maximum(-advantages * ratio, -advantages * clipped_ratio)
    loss = aggregate_loss(pg_losses, response_mask, loss_agg_mode)
    return ProsLossStats(
        loss=loss,
        pg_loss=loss,
        kl_loss=torch.zeros((), device=loss.device, dtype=loss.dtype),
        approx_kl=masked_mean(-negative_approx_kl, response_mask),
        clip_fraction=masked_mean((clipped_ratio != ratio).to(log_prob.dtype), response_mask),
        advantage_mean=masked_mean(advantages, response_mask),
        advantage_abs_mean=masked_mean(advantages.abs(), response_mask),
    )


def compute_gpg_policy_loss(log_prob, advantages, response_mask, *, loss_agg_mode: str):
    """Unclipped GPG policy gradient loss from the reference PROS code."""

    torch = _torch()
    pg_losses = -log_prob * advantages
    loss = aggregate_loss(pg_losses, response_mask, loss_agg_mode)
    zero = torch.zeros((), device=loss.device, dtype=loss.dtype)
    return ProsLossStats(
        loss=loss,
        pg_loss=loss,
        kl_loss=zero,
        approx_kl=zero,
        clip_fraction=zero,
        advantage_mean=masked_mean(advantages, response_mask),
        advantage_abs_mean=masked_mean(advantages.abs(), response_mask),
    )


def compute_policy_loss(
    *,
    objective: Literal["fastgrpo", "clipped_grpo", "gpg"],
    log_prob,
    old_log_prob,
    ref_log_prob,
    advantages,
    response_mask,
    epsilon: float,
    beta: float,
    loss_agg_mode: str,
):
    if objective == "fastgrpo":
        return compute_fastgrpo_policy_loss(
            log_prob,
            old_log_prob,
            ref_log_prob,
            advantages,
            response_mask,
            epsilon=epsilon,
            beta=beta,
            loss_agg_mode=loss_agg_mode,
        )
    if objective == "clipped_grpo":
        return compute_clipped_grpo_policy_loss(
            log_prob,
            old_log_prob,
            advantages,
            response_mask,
            epsilon=epsilon,
            loss_agg_mode=loss_agg_mode,
        )
    if objective == "gpg":
        return compute_gpg_policy_loss(log_prob, advantages, response_mask, loss_agg_mode=loss_agg_mode)
    raise ValueError(f"Unknown PROS objective: {objective}")
