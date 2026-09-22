"""Typed trace and privileged-information records used by TASPO."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class StepTrace:
    step: int
    pre_observation: str
    action: str
    post_observation: str
    is_action_valid: bool
    done: bool


@dataclass(frozen=True)
class TrajectoryTrace:
    group_uid: str
    traj_uid: str
    task: str
    verified_success: float
    episode_reward: float
    steps: tuple[StepTrace, ...]


@dataclass(frozen=True)
class SourceRef:
    traj_uid: str
    step: int
    pre_quote: str
    action_quote: str
    post_quote: str


@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    kind: str
    statement: str
    sources: tuple[SourceRef, ...]


@dataclass(frozen=True)
class TargetSupport:
    step: int
    field: str
    quote: str


@dataclass(frozen=True)
class GuidanceItem:
    statement: str
    evidence_ids: tuple[str, ...]
    target_support: tuple[TargetSupport, ...]


@dataclass
class TrajectoryPI:
    traj_uid: str
    teacher_pi: str = ""
    guidance: list[GuidanceItem] = field(default_factory=list)
    allowed_evidence_ids: list[str] = field(default_factory=list)
    abstain_reason: str = ""

    @property
    def available(self) -> bool:
        return bool(self.teacher_pi.strip())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
