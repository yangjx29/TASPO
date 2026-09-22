import tempfile
import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from agent_system.taspo.analyzer import TASPOAnalyzer
from tests.taspo.test_analyzer import GoodClient, make_batch


@unittest.skipIf(torch is None, "PyTorch is required for the TASPO pipeline test")
class PipelineTest(unittest.TestCase):
    def test_aligned_pi_flows_into_sign_preserving_assignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            analyzer = TASPOAnalyzer(
                {
                    "enabled": True,
                    "max_workers": 1,
                    "save_audit": True,
                    "audit_dir": tmp,
                    "skip_uniform_reward_groups": False,
                },
                client=GoodClient(),
            )
            batch, _ = analyzer.annotate_batch(make_batch(), global_step=1)

        traj_uids = batch.non_tensor_batch["traj_uid"]
        success_rows = traj_uids == "success"
        failure_rows = traj_uids == "failure"
        self.assertFalse(np.any(batch.non_tensor_batch["taspo_pi_available"][success_rows]))
        self.assertTrue(np.all(batch.non_tensor_batch["taspo_pi_available"][failure_rows]))

        from agent_system.taspo.credit import build_taspo_advantages

        advantages = torch.tensor([[1.0], [-1.0], [-1.0]])
        student = torch.zeros_like(advantages)
        teacher = torch.tensor([[0.2], [0.5], [-0.8]])
        mask = torch.ones_like(advantages)
        shaped, tensors, metrics = build_taspo_advantages(
            advantages,
            student,
            teacher,
            mask,
            traj_uids,
            batch.non_tensor_batch["turn_step"],
            batch.non_tensor_batch["taspo_pi_available"],
            action_mask=mask,
            trajectory_balance=False,
        )

        self.assertEqual(tensors["action_weight"][success_rows].item(), 1.0)
        self.assertAlmostEqual(
            tensors["action_weight"][failure_rows].mean().item(),
            1.0,
            places=6,
        )
        self.assertTrue(torch.all(shaped[failure_rows] < 0))
        self.assertEqual(metrics["taspo/outcome_sign_flips"], 0.0)


if __name__ == "__main__":
    unittest.main()
