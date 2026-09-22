import tempfile
import unittest

import numpy as np

from agent_system.taspo.analyzer import TASPOAnalyzer


class FakeBatch:
    def __init__(self, rows):
        self.non_tensor_batch = {key: np.array([row[key] for row in rows], dtype=object) for key in rows[0]}

    def __len__(self):
        return len(self.non_tensor_batch["traj_uid"])


def make_batch():
    rows = []
    for traj_uid, success, actions in [
        ("success", 1.0, [("kitchen", "take apple", "holding apple")]),
        (
            "failure",
            0.0,
            [
                ("kitchen", "take knife", "holding knife"),
                ("microwave closed", "put knife in microwave", "knife in microwave"),
            ],
        ),
    ]:
        for step, (pre, action, post) in enumerate(actions):
            rows.append(
                {
                    "uid": "group-1",
                    "traj_uid": traj_uid,
                    "turn_step": step,
                    "task_text": "heat the apple and place it on the table",
                    "verified_success": success,
                    "episode_rewards": success,
                    "pre_observation": pre,
                    "action_text": action,
                    "post_observation": post,
                    "is_action_valid": True,
                    "env_done": step == len(actions) - 1,
                }
            )
    return FakeBatch(rows)


class GoodClient:
    def __init__(self):
        self.calls = []

    def complete_json(self, _system, payload, namespace):
        self.calls.append((namespace, payload))
        if namespace == "source":
            return {
                "evidence": [
                    {
                        "evidence_id": "E1",
                        "kind": "progress",
                        "statement": "Select the apple before attempting to heat it.",
                        "sources": [
                            {
                                "traj_uid": "success",
                                "step": 0,
                                "pre_quote": "kitchen",
                                "action_quote": "take apple",
                                "post_quote": "holding apple",
                            }
                        ],
                    }
                ]
            }
        assert payload["targets"][0]["traj_uid"] == "failure"
        assert payload["targets"][0]["allowed_evidence_ids"] == ["E1"]
        return {
            "targets": [
                {
                    "traj_uid": "failure",
                    "guidance": [
                        {
                            "statement": "Taking the knife does not advance the apple-heating task.",
                            "evidence_ids": ["E1"],
                            "target_support": [{"step": 0, "field": "action", "quote": "take knife"}],
                        }
                    ],
                    "abstain_reason": "",
                }
            ]
        }


class InvalidEvidenceClient(GoodClient):
    def complete_json(self, _system, payload, namespace):
        self.calls.append((namespace, payload))
        return {
            "evidence": [
                {
                    "evidence_id": "E1",
                    "kind": "progress",
                    "statement": "unsupported",
                    "sources": [
                        {
                            "traj_uid": "success",
                            "step": 0,
                            "pre_quote": "invented room",
                            "action_quote": "invented action",
                            "post_quote": "invented state",
                        }
                    ],
                }
            ]
        }


class LeaveOneOutClient:
    def __init__(self):
        self.align_payload = None

    def complete_json(self, _system, payload, namespace):
        if namespace == "source":
            return {
                "evidence": [
                    {
                        "evidence_id": "E1",
                        "kind": "alternative",
                        "statement": "Either verified action can make progress.",
                        "sources": [
                            {
                                "traj_uid": "s1",
                                "step": 0,
                                "pre_quote": "state one",
                                "action_quote": "action one",
                                "post_quote": "progress one",
                            },
                            {
                                "traj_uid": "s2",
                                "step": 0,
                                "pre_quote": "state two",
                                "action_quote": "action two",
                                "post_quote": "progress two",
                            },
                        ],
                    }
                ]
            }
        self.align_payload = payload
        return {"targets": []}


class AnalyzerTest(unittest.TestCase):
    def config(self, audit_dir):
        return {
            "enabled": True,
            "required": False,
            "max_workers": 1,
            "save_audit": True,
            "audit_dir": audit_dir,
            "max_trace_chars": 4000,
            "skip_uniform_reward_groups": False,
        }

    def test_alignment_is_fixed_per_target_and_leave_one_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = GoodClient()
            analyzer = TASPOAnalyzer(self.config(tmp), client=client)
            batch, metrics = analyzer.annotate_batch(make_batch(), global_step=3)

            failure = batch.non_tensor_batch["traj_uid"] == "failure"
            success = batch.non_tensor_batch["traj_uid"] == "success"
            failure_pi = batch.non_tensor_batch["taspo_teacher_pi"][failure]
            self.assertTrue(np.all(failure_pi == failure_pi[0]))
            self.assertTrue(failure_pi[0].startswith("- Taking the knife"))
            self.assertFalse(np.any(batch.non_tensor_batch["taspo_pi_available"][success]))
            self.assertEqual([call[0] for call in client.calls], ["source", "align"])
            self.assertAlmostEqual(metrics["taspo/pi_trajectory_coverage"], 0.5)

    def test_unsupported_source_quotes_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = InvalidEvidenceClient()
            analyzer = TASPOAnalyzer(self.config(tmp), client=client)
            batch, metrics = analyzer.annotate_batch(make_batch(), global_step=1)

            self.assertFalse(np.any(batch.non_tensor_batch["taspo_pi_available"]))
            self.assertEqual([call[0] for call in client.calls], ["source"])
            self.assertEqual(metrics["taspo/pi_trajectory_coverage"], 0.0)

    def test_invalid_action_cannot_become_source_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = GoodClient()
            batch = make_batch()
            success = batch.non_tensor_batch["traj_uid"] == "success"
            batch.non_tensor_batch["is_action_valid"][success] = False
            analyzer = TASPOAnalyzer(self.config(tmp), client=client)
            batch, metrics = analyzer.annotate_batch(batch, global_step=1)

            self.assertFalse(np.any(batch.non_tensor_batch["taspo_pi_available"]))
            self.assertEqual([call[0] for call in client.calls], ["source"])
            self.assertEqual(metrics["taspo/pi_trajectory_coverage"], 0.0)

    def test_disabled_analyzer_is_exact_abstention(self):
        analyzer = TASPOAnalyzer({"enabled": False})
        batch, _ = analyzer.annotate_batch(make_batch(), global_step=1)
        self.assertFalse(np.any(batch.non_tensor_batch["taspo_pi_available"]))
        self.assertTrue(np.all(batch.non_tensor_batch["taspo_teacher_pi"] == ""))

    def test_leave_one_out_removes_target_source_quotes(self):
        rows = []
        for traj_uid, pre, action, post in [
            ("s1", "state one", "action one", "progress one"),
            ("s2", "state two", "action two", "progress two"),
        ]:
            rows.append(
                {
                    "uid": "group-loo",
                    "traj_uid": traj_uid,
                    "turn_step": 0,
                    "task_text": "task",
                    "verified_success": 1.0,
                    "episode_rewards": 1.0,
                    "pre_observation": pre,
                    "action_text": action,
                    "post_observation": post,
                    "is_action_valid": True,
                    "env_done": True,
                }
            )
        with tempfile.TemporaryDirectory() as tmp:
            client = LeaveOneOutClient()
            analyzer = TASPOAnalyzer(self.config(tmp), client=client)
            analyzer.annotate_batch(FakeBatch(rows), global_step=1)

        targets = {item["traj_uid"]: item for item in client.align_payload["targets"]}
        s1_support = targets["s1"]["allowed_evidence"][0]["source_support"]
        s2_support = targets["s2"]["allowed_evidence"][0]["source_support"]
        self.assertEqual([item["action_quote"] for item in s1_support], ["action two"])
        self.assertEqual([item["action_quote"] for item in s2_support], ["action one"])


if __name__ == "__main__":
    unittest.main()
