"""Outcome-anchored action credit allocation for TASPO."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch


def _masked_row_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denominator = mask.sum(dim=-1).clamp_min(1)
    return (values * mask).sum(dim=-1) / denominator


def build_taspo_advantages(
    seq_advantages: torch.Tensor,
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    traj_uids: np.ndarray,
    turn_steps: np.ndarray,
    pi_available: np.ndarray | torch.Tensor,
    action_mask: torch.Tensor | None = None,
    padding_rows: np.ndarray | torch.Tensor | None = None,
    epsilon: float = 0.4,
    temperature: float = 0.5,
    token_gap_clip: float = 2.0,
    trajectory_balance: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, float]]:
    """Allocate outcome advantage across actions without changing its sign.

    Teacher-policy shifts are averaged over tokens within each action. Their
    outcome-aligned, bounded values are centered within each trajectory, so the
    mean surgical weight is exactly one. A separate multiplier implements
    action-token -> trajectory-step -> batch-trajectory aggregation when the
    actor uses ``seq-mean-token-mean``.
    """
    if not 0.0 <= epsilon < 1.0:
        raise ValueError("epsilon must lie in [0, 1) so action weights stay positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if token_gap_clip <= 0:
        raise ValueError("token_gap_clip must be positive")
    if seq_advantages.shape != student_log_probs.shape:
        raise ValueError("seq_advantages and student_log_probs must have identical shapes")
    if teacher_log_probs.shape != student_log_probs.shape or response_mask.shape != student_log_probs.shape:
        raise ValueError("teacher/student log probs and response_mask must have identical shapes")

    device = seq_advantages.device
    dtype = seq_advantages.dtype
    batch_size = seq_advantages.size(0)
    mask = response_mask.to(dtype)
    score_mask = mask if action_mask is None else action_mask.to(device=device, dtype=dtype)
    if score_mask.shape != mask.shape:
        raise ValueError("action_mask must have the same shape as response_mask")
    score_mask = score_mask * mask
    if isinstance(pi_available, torch.Tensor):
        pi_mask = pi_available.to(device=device, dtype=torch.bool)
    else:
        pi_mask = torch.as_tensor(pi_available, device=device, dtype=torch.bool)
    if padding_rows is None:
        padding_mask = torch.zeros(batch_size, device=device, dtype=torch.bool)
    elif isinstance(padding_rows, torch.Tensor):
        padding_mask = padding_rows.to(device=device, dtype=torch.bool)
    else:
        padding_mask = torch.as_tensor(padding_rows, device=device, dtype=torch.bool)

    finite_log_probs = torch.isfinite(teacher_log_probs) & torch.isfinite(student_log_probs)
    score_mask = score_mask * finite_log_probs.to(dtype)
    pi_mask = pi_mask & (score_mask.sum(dim=-1) > 0)

    # TASPO-style credit allocation: compute an executable-action mean from the
    # raw clean/PI likelihood shift, center that raw score within a trajectory,
    # then apply outcome-sign alignment and tanh. token_gap_clip is retained in
    # the public signature for launcher compatibility but is intentionally not
    # used by this allocation rule.
    token_shift = torch.where(
        finite_log_probs,
        (teacher_log_probs - student_log_probs).detach(),
        torch.zeros_like(student_log_probs),
    )
    step_score = _masked_row_mean(token_shift, score_mask)
    row_advantage = _masked_row_mean(seq_advantages, mask)
    if not torch.all(torch.isfinite(row_advantage[~padding_mask])):
        raise RuntimeError("Non-finite outcome advantage encountered before TASPO allocation")
    centered_step_score = torch.zeros(batch_size, device=device, dtype=dtype)
    reward_aligned_score = torch.zeros(batch_size, device=device, dtype=dtype)

    groups: dict[str, list[int]] = defaultdict(list)
    for index, traj_uid in enumerate(traj_uids):
        if not bool(padding_mask[index]):
            groups[str(traj_uid)].append(index)
    for indices in groups.values():
        indices.sort(key=lambda index: int(turn_steps[index]))

    action_weight = torch.ones(batch_size, device=device, dtype=dtype)
    balance_weight = torch.ones(batch_size, device=device, dtype=dtype)
    conservation_errors = []
    active_pi_trajectories = 0
    num_trajectories = len(groups)
    total_rows = batch_size

    for indices in groups.values():
        index_tensor = torch.as_tensor(indices, device=device, dtype=torch.long)
        # Retain TASPO's trajectory-level fixed-PI and fail-closed eligibility:
        # all steps must have usable PI before any step receives a credit shift.
        trajectory_has_pi = bool(torch.all(pi_mask[index_tensor]).item())
        trajectory_advantages = row_advantage[index_tensor]
        nonzero_advantages = trajectory_advantages[trajectory_advantages != 0]
        if nonzero_advantages.numel() and not torch.all(
            torch.sign(nonzero_advantages) == torch.sign(nonzero_advantages[0])
        ):
            raise RuntimeError("A trajectory contains inconsistent outcome-advantage signs")
        if trajectory_has_pi and nonzero_advantages.numel():
            centered = step_score[index_tensor] - step_score[index_tensor].mean()
            centered_step_score[index_tensor] = centered
            outcome_sign = torch.sign(nonzero_advantages[0])
            q = torch.tanh(outcome_sign * centered / temperature)
            reward_aligned_score[index_tensor] = q
            local_weight = 1.0 + (epsilon / 2.0) * (q - q.mean())
            action_weight[index_tensor] = local_weight
            conservation_errors.append(abs(local_weight.mean().item() - 1.0))
            active_pi_trajectories += 1
        else:
            conservation_errors.append(0.0)

        if trajectory_balance:
            balance_weight[index_tensor] = total_rows / max(1, num_trajectories * len(indices))

    if not trajectory_balance:
        valid_count = int((~padding_mask).sum().item())
        if valid_count:
            balance_weight[~padding_mask] = total_rows / valid_count

    action_weight[padding_mask] = 0.0
    balance_weight[padding_mask] = 0.0
    combined_weight = action_weight * balance_weight
    shaped_advantages = seq_advantages * combined_weight.unsqueeze(-1)

    valid_rows = ~padding_mask
    old_sign = torch.sign(row_advantage[valid_rows])
    new_row_advantage = _masked_row_mean(shaped_advantages, mask)[valid_rows]
    new_sign = torch.sign(new_row_advantage)
    nonzero = old_sign != 0
    sign_flips = ((old_sign != new_sign) & nonzero).float().sum().item()

    tensors = {
        "step_score": step_score,
        "centered_step_score": centered_step_score,
        "reward_aligned_score": reward_aligned_score,
        "action_weight": action_weight,
        "trajectory_balance": balance_weight,
        "combined_weight": combined_weight,
    }
    valid_action_weights = action_weight[valid_rows]
    metrics = {
        "taspo/step_score_mean": step_score[valid_rows].mean().item() if valid_rows.any() else 0.0,
        "taspo/step_score_std": step_score[valid_rows].std(unbiased=False).item() if valid_rows.any() else 0.0,
        "taspo/pi_step_score_mean": step_score[pi_mask & valid_rows].mean().item() if (pi_mask & valid_rows).any() else 0.0,
        "taspo/action_weight_mean": valid_action_weights.mean().item() if valid_rows.any() else 0.0,
        "taspo/action_weight_std": valid_action_weights.std(unbiased=False).item() if valid_rows.any() else 0.0,
        "taspo/action_weight_min": valid_action_weights.min().item() if valid_rows.any() else 0.0,
        "taspo/action_weight_max": valid_action_weights.max().item() if valid_rows.any() else 0.0,
        "taspo/credit_conservation_max_error": max(conservation_errors, default=0.0),
        "taspo/outcome_sign_flips": float(sign_flips),
        "taspo/active_pi_trajectory_ratio": active_pi_trajectories / max(1, num_trajectories),
        "taspo/padding_row_ratio": padding_mask.float().mean().item(),
    }
    return shaped_advantages, tensors, metrics
