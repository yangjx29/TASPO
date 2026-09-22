"""Reconstruct compact, auditable trajectories from a rollout DataProto."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from .schema import StepTrace, TrajectoryTrace


def normalize_text(value: object) -> str:
    return " ".join(str(value or "").split())


def truncate_middle(text: str, limit: int) -> str:
    text = normalize_text(text)
    if limit <= 0 or len(text) <= limit:
        return text
    if limit < 16:
        return text[:limit]
    left = (limit - 5) // 2
    right = limit - 5 - left
    return f"{text[:left]} ... {text[-right:]}"


def _value(batch, key: str, index: int, default=""):
    values = batch.non_tensor_batch.get(key)
    if values is None:
        return default
    return values[index]


def build_trajectory_groups(batch) -> dict[str, list[TrajectoryTrace]]:
    """Group rollout rows by prompt group and trajectory, ordered by turn."""
    rows: dict[tuple[str, str], list[int]] = defaultdict(list)
    group_for_traj: dict[str, str] = {}

    for index in range(len(batch)):
        if bool(_value(batch, "is_padding_row", index, False)):
            continue
        group_uid = str(_value(batch, "uid", index))
        traj_uid = str(_value(batch, "traj_uid", index))
        rows[(group_uid, traj_uid)].append(index)
        group_for_traj[traj_uid] = group_uid

    groups: dict[str, list[TrajectoryTrace]] = defaultdict(list)
    for (group_uid, traj_uid), indices in rows.items():
        indices.sort(key=lambda idx: int(_value(batch, "turn_step", idx, 0)))
        first = indices[0]
        steps = tuple(
            StepTrace(
                step=int(_value(batch, "turn_step", idx, 0)),
                pre_observation=str(_value(batch, "pre_observation", idx, "")),
                action=str(_value(batch, "action_text", idx, "")),
                post_observation=str(_value(batch, "post_observation", idx, "")),
                is_action_valid=bool(_value(batch, "is_action_valid", idx, True)),
                done=bool(_value(batch, "env_done", idx, False)),
            )
            for idx in indices
        )
        groups[group_uid].append(
            TrajectoryTrace(
                group_uid=group_uid,
                traj_uid=traj_uid,
                task=str(_value(batch, "task_text", first, "")),
                verified_success=float(_value(batch, "verified_success", first, 0.0)),
                episode_reward=float(_value(batch, "episode_rewards", first, 0.0)),
                steps=steps,
            )
        )

    for trajectories in groups.values():
        trajectories.sort(key=lambda trace: trace.traj_uid)
    return dict(groups)


def compact_trace(
    trace: TrajectoryTrace,
    max_trace_chars: int = 12_000,
    max_action_chars: int = 320,
) -> dict:
    """Bound analyzer context while retaining evidence from every environment step."""
    step_count = max(1, len(trace.steps))
    task_budget = min(1_200, max(300, max_trace_chars // 10))
    remaining = max(1_000, max_trace_chars - task_budget)
    per_step = max(180, remaining // step_count)
    action_budget = min(max_action_chars, max(80, per_step // 3))
    observation_budget = max(40, (per_step - action_budget) // 2)

    return {
        "traj_uid": trace.traj_uid,
        "task": truncate_middle(trace.task, task_budget),
        "verified_success": trace.verified_success,
        "episode_reward": trace.episode_reward,
        "steps": [
            {
                "step": step.step,
                "pre": truncate_middle(step.pre_observation, observation_budget),
                "action": truncate_middle(step.action, action_budget),
                "post": truncate_middle(step.post_observation, observation_budget),
                "valid": step.is_action_valid,
                "done": step.done,
            }
            for step in trace.steps
        ],
    }


def find_step(trace: TrajectoryTrace, step_index: int) -> StepTrace | None:
    for step in trace.steps:
        if step.step == step_index:
            return step
    return None


def quote_is_supported(quote: str, containers: Iterable[str]) -> bool:
    normalized_quote = normalize_text(quote)
    if not normalized_quote:
        return False
    return any(normalized_quote in normalize_text(container) for container in containers)
