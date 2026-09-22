"""Two-stage, evidence-grounded PI construction for TASPO."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .client import AnalyzerClientConfig, OpenAIJSONClient
from .prompts import ALIGN_SYSTEM_PROMPT, SOURCE_SYSTEM_PROMPT
from .schema import (
    EvidenceItem,
    GuidanceItem,
    SourceRef,
    TargetSupport,
    TrajectoryPI,
    TrajectoryTrace,
)
from .trace import build_trajectory_groups, compact_trace, find_step, quote_is_supported

logger = logging.getLogger(__name__)


def _cfg_get(config, key: str, default=None):
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


class TASPOAnalyzer:
    """Build one fixed, trajectory-specific PI record for every rollout."""

    def __init__(self, config, client=None):
        self.config = config
        self.enabled = bool(_cfg_get(config, "enabled", True))
        self.required = bool(_cfg_get(config, "required", False))
        self.success_threshold = float(_cfg_get(config, "success_threshold", 0.5))
        self.max_trace_chars = int(_cfg_get(config, "max_trace_chars", 12_000))
        self.max_guidance_items = int(_cfg_get(config, "max_guidance_items", 8))
        self.max_workers = int(_cfg_get(config, "max_workers", 4))
        self.skip_uniform_reward_groups = bool(_cfg_get(config, "skip_uniform_reward_groups", True))
        self.audit_dir = Path(str(_cfg_get(config, "audit_dir", "outputs/taspo_audit"))).expanduser()
        self.save_audit = bool(_cfg_get(config, "save_audit", True))
        self._client = client
        if self.enabled and self._client is None:
            self._client = OpenAIJSONClient(AnalyzerClientConfig.from_config(config))

    def annotate_batch(self, batch, global_step: int) -> tuple[object, dict[str, float]]:
        usage_before = self._client.snapshot_usage() if self._client is not None and hasattr(self._client, "snapshot_usage") else {}
        groups = build_trajectory_groups(batch)
        trajectory_count = sum(len(group) for group in groups.values())
        results: dict[str, TrajectoryPI] = {}
        audits: list[dict] = []
        failures = 0
        source_calls = 0
        align_calls = 0

        if self.enabled and groups:
            with ThreadPoolExecutor(max_workers=max(1, self.max_workers)) as executor:
                futures = {executor.submit(self._analyze_group, group_uid, trajectories): group_uid for group_uid, trajectories in groups.items()}
                for future in as_completed(futures):
                    group_uid = futures[future]
                    try:
                        group_results, audit, calls = future.result()
                        results.update(group_results)
                        audits.append(audit)
                        source_calls += calls[0]
                        align_calls += calls[1]
                    except Exception as exc:
                        failures += 1
                        logger.exception("TASPO analyzer failed for group %s", group_uid)
                        if self.required:
                            raise
                        audits.append(
                            {
                                "group_uid": group_uid,
                                "error": f"{type(exc).__name__}: {exc}",
                                "evidence": [],
                                "targets": {},
                            }
                        )
                        for trace in groups[group_uid]:
                            results[trace.traj_uid] = TrajectoryPI(
                                traj_uid=trace.traj_uid,
                                abstain_reason=f"analyzer_error:{type(exc).__name__}",
                            )

        for trajectories in groups.values():
            for trace in trajectories:
                results.setdefault(
                    trace.traj_uid,
                    TrajectoryPI(
                        traj_uid=trace.traj_uid,
                        abstain_reason="analyzer_disabled" if not self.enabled else "no_verified_evidence",
                    ),
                )

        row_pi = []
        row_available = []
        row_reason = []
        row_guidance_count = []
        for index in range(len(batch)):
            traj_uid = str(batch.non_tensor_batch["traj_uid"][index])
            item = results.get(traj_uid, TrajectoryPI(traj_uid=traj_uid, abstain_reason="unmapped"))
            row_pi.append(item.teacher_pi)
            row_available.append(item.available)
            row_reason.append(item.abstain_reason)
            row_guidance_count.append(len(item.guidance))

        batch.non_tensor_batch["taspo_teacher_pi"] = np.array(row_pi, dtype=object)
        batch.non_tensor_batch["taspo_pi_available"] = np.array(row_available, dtype=bool)
        batch.non_tensor_batch["taspo_pi_reason"] = np.array(row_reason, dtype=object)
        batch.non_tensor_batch["taspo_guidance_count"] = np.array(row_guidance_count, dtype=np.int64)

        if self.save_audit and audits:
            self.audit_dir.mkdir(parents=True, exist_ok=True)
            path = self.audit_dir / f"step_{global_step:06d}.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for audit in sorted(audits, key=lambda item: item["group_uid"]):
                    handle.write(json.dumps(audit, ensure_ascii=False) + "\n")

        available_trajectories = sum(item.available for item in results.values())
        metrics = {
            "taspo/pi_trajectory_coverage": available_trajectories / max(1, trajectory_count),
            "taspo/pi_row_coverage": float(np.mean(row_available)) if row_available else 0.0,
            "taspo/analyzer_group_failures": float(failures),
            "taspo/analyzer_source_calls": float(source_calls),
            "taspo/analyzer_align_calls": float(align_calls),
        }
        if usage_before:
            usage_after = self._client.snapshot_usage()
            for key in ["live_calls", "cache_hits", "prompt_tokens", "completion_tokens"]:
                metrics[f"taspo/analyzer_{key}"] = float(usage_after.get(key, 0) - usage_before.get(key, 0))
        return batch, metrics

    def _analyze_group(
        self,
        group_uid: str,
        trajectories: list[TrajectoryTrace],
    ) -> tuple[dict[str, TrajectoryPI], dict, tuple[int, int]]:
        if self.skip_uniform_reward_groups:
            rewards = [trace.episode_reward for trace in trajectories]
            if rewards and max(rewards) - min(rewards) <= 1e-8:
                empty = {
                    trace.traj_uid: TrajectoryPI(
                        trace.traj_uid,
                        abstain_reason="uniform_outcome_group",
                    )
                    for trace in trajectories
                }
                return (
                    empty,
                    {"group_uid": group_uid, "evidence": [], "targets": {}},
                    (0, 0),
                )
        successful = [trace for trace in trajectories if trace.verified_success >= self.success_threshold]
        empty = {trace.traj_uid: TrajectoryPI(trace.traj_uid, abstain_reason="no_verified_success_sibling") for trace in trajectories}
        if not successful:
            return empty, {"group_uid": group_uid, "evidence": [], "targets": {}}, (0, 0)

        source_payload = {
            "group_uid": group_uid,
            "successful_trajectories": [compact_trace(trace, self.max_trace_chars) for trace in successful],
        }
        source_raw = self._client.complete_json(SOURCE_SYSTEM_PROMPT, source_payload, "source")
        evidence = self._validate_evidence(source_raw, successful)
        if not evidence:
            return empty, {"group_uid": group_uid, "evidence": [], "targets": {}}, (1, 0)

        allowed_by_target: dict[str, list[str]] = {}
        evidence_by_id = {item.evidence_id: item for item in evidence}
        for target in trajectories:
            allowed_by_target[target.traj_uid] = [item.evidence_id for item in evidence if any(source.traj_uid != target.traj_uid for source in item.sources)]

        targets_with_evidence = [trace for trace in trajectories if allowed_by_target[trace.traj_uid]]
        if not targets_with_evidence:
            return empty, self._audit(group_uid, evidence, empty), (1, 0)

        align_payload = {
            "group_uid": group_uid,
            "targets": [
                {
                    **compact_trace(trace, self.max_trace_chars),
                    "allowed_evidence_ids": allowed_by_target[trace.traj_uid],
                    "allowed_evidence": [
                        {
                            "evidence_id": item.evidence_id,
                            "kind": item.kind,
                            "statement": item.statement,
                            "source_support": [
                                {
                                    "pre_quote": source.pre_quote,
                                    "action_quote": source.action_quote,
                                    "post_quote": source.post_quote,
                                }
                                for source in item.sources
                                if source.traj_uid != trace.traj_uid
                            ],
                        }
                        for item in evidence
                        if item.evidence_id in allowed_by_target[trace.traj_uid]
                    ],
                }
                for trace in targets_with_evidence
            ],
        }
        align_raw = self._client.complete_json(ALIGN_SYSTEM_PROMPT, align_payload, "align")
        results = self._validate_alignment(
            align_raw,
            trajectories,
            evidence_by_id,
            allowed_by_target,
        )
        for trace in trajectories:
            results.setdefault(
                trace.traj_uid,
                TrajectoryPI(
                    traj_uid=trace.traj_uid,
                    allowed_evidence_ids=allowed_by_target[trace.traj_uid],
                    abstain_reason="no_leave_one_out_evidence" if not allowed_by_target[trace.traj_uid] else "alignment_abstained",
                ),
            )
        return results, self._audit(group_uid, evidence, results), (1, 1)

    def _validate_evidence(
        self,
        raw: dict,
        successful: list[TrajectoryTrace],
    ) -> list[EvidenceItem]:
        traces = {trace.traj_uid: trace for trace in successful}
        valid: list[EvidenceItem] = []
        seen_ids: set[str] = set()
        for candidate in raw.get("evidence", []):
            if not isinstance(candidate, dict):
                continue
            evidence_id = str(candidate.get("evidence_id", "")).strip()
            statement = str(candidate.get("statement", "")).strip()
            kind = str(candidate.get("kind", "progress")).strip()
            if not evidence_id or evidence_id in seen_ids or not statement or len(statement) > 800:
                continue
            refs: list[SourceRef] = []
            for source in candidate.get("sources", []):
                if not isinstance(source, dict):
                    continue
                traj_uid = str(source.get("traj_uid", ""))
                trace = traces.get(traj_uid)
                step = find_step(trace, int(source.get("step", -1))) if trace else None
                if step is None or not step.is_action_valid:
                    continue
                pre_quote = str(source.get("pre_quote", ""))
                action_quote = str(source.get("action_quote", ""))
                post_quote = str(source.get("post_quote", ""))
                action_ok = quote_is_supported(action_quote, [step.action])
                state_ok = quote_is_supported(
                    pre_quote,
                    [step.pre_observation],
                ) or quote_is_supported(post_quote, [step.post_observation])
                if action_ok and state_ok:
                    refs.append(
                        SourceRef(
                            traj_uid=traj_uid,
                            step=step.step,
                            pre_quote=pre_quote,
                            action_quote=action_quote,
                            post_quote=post_quote,
                        )
                    )
            if refs:
                seen_ids.add(evidence_id)
                valid.append(EvidenceItem(evidence_id, kind, statement, tuple(refs)))
        return valid

    def _validate_alignment(
        self,
        raw: dict,
        trajectories: list[TrajectoryTrace],
        evidence_by_id: dict[str, EvidenceItem],
        allowed_by_target: dict[str, list[str]],
    ) -> dict[str, TrajectoryPI]:
        traces = {trace.traj_uid: trace for trace in trajectories}
        results: dict[str, TrajectoryPI] = {}
        for target_raw in raw.get("targets", []):
            if not isinstance(target_raw, dict):
                continue
            traj_uid = str(target_raw.get("traj_uid", ""))
            trace = traces.get(traj_uid)
            if trace is None or traj_uid in results:
                continue
            allowed = set(allowed_by_target.get(traj_uid, []))
            guidance: list[GuidanceItem] = []
            for item_raw in target_raw.get("guidance", [])[: self.max_guidance_items]:
                if not isinstance(item_raw, dict):
                    continue
                statement = str(item_raw.get("statement", "")).strip()
                evidence_ids = tuple(dict.fromkeys(str(item) for item in item_raw.get("evidence_ids", [])))
                if not statement or len(statement) > 800 or not evidence_ids or any(item not in allowed or item not in evidence_by_id for item in evidence_ids):
                    continue
                supports: list[TargetSupport] = []
                for support_raw in item_raw.get("target_support", []):
                    if not isinstance(support_raw, dict):
                        continue
                    field = str(support_raw.get("field", ""))
                    quote = str(support_raw.get("quote", ""))
                    step_index = int(support_raw.get("step", -1))
                    if field == "task":
                        supported = quote_is_supported(quote, [trace.task])
                    else:
                        step = find_step(trace, step_index)
                        values = {
                            "pre": [step.pre_observation] if step else [],
                            "action": [step.action] if step else [],
                            "post": [step.post_observation] if step else [],
                        }.get(field, [])
                        supported = quote_is_supported(quote, values)
                    if supported:
                        supports.append(TargetSupport(step_index, field, quote))
                if supports:
                    guidance.append(GuidanceItem(statement, evidence_ids, tuple(supports)))

            teacher_pi = ""
            if guidance:
                teacher_pi = "\n".join(f"- {item.statement}" for item in guidance)
            results[traj_uid] = TrajectoryPI(
                traj_uid=traj_uid,
                teacher_pi=teacher_pi,
                guidance=guidance,
                allowed_evidence_ids=sorted(allowed),
                abstain_reason="" if guidance else str(target_raw.get("abstain_reason", "alignment_abstained")),
            )
        return results

    @staticmethod
    def _audit(
        group_uid: str,
        evidence: list[EvidenceItem],
        results: dict[str, TrajectoryPI],
    ) -> dict:
        return {
            "group_uid": group_uid,
            "evidence": [
                {
                    "evidence_id": item.evidence_id,
                    "kind": item.kind,
                    "statement": item.statement,
                    "sources": [source.__dict__ for source in item.sources],
                }
                for item in evidence
            ],
            "targets": {traj_uid: result.to_dict() for traj_uid, result in results.items()},
        }
