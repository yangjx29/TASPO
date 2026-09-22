import unittest

import numpy as np

try:
    import torch
except ImportError:  # Local documentation machines may not include the GPU stack.
    torch = None


@unittest.skipIf(torch is None, "PyTorch is required for TASPO credit tests")
class CreditTest(unittest.TestCase):
    def test_failure_weights_are_bounded_conserved_and_sign_preserving(self):
        from agent_system.taspo.credit import build_taspo_advantages

        advantages = -torch.ones(4, 1)
        student = torch.zeros(4, 1)
        teacher = torch.tensor([[0.6], [0.0], [-0.8], [-0.2]])
        mask = torch.ones_like(advantages)
        shaped, tensors, metrics = build_taspo_advantages(
            advantages,
            student,
            teacher,
            mask,
            np.array(["t"] * 4),
            np.arange(4),
            np.ones(4, dtype=bool),
            epsilon=0.4,
            temperature=1.0,
            trajectory_balance=False,
        )

        weights = tensors["action_weight"]
        self.assertAlmostEqual(weights.mean().item(), 1.0, places=6)
        self.assertGreaterEqual(weights.min().item(), 0.6)
        self.assertLessEqual(weights.max().item(), 1.4)
        self.assertLess(weights[0].item(), weights[2].item())
        self.assertTrue(torch.all(shaped < 0))
        self.assertEqual(metrics["taspo/outcome_sign_flips"], 0.0)

    def test_empty_pi_is_identity_without_trajectory_balancing(self):
        from agent_system.taspo.credit import build_taspo_advantages

        advantages = torch.tensor([[2.0, 2.0], [-1.0, -1.0]])
        zeros = torch.zeros_like(advantages)
        mask = torch.ones_like(advantages)
        shaped, tensors, _ = build_taspo_advantages(
            advantages,
            zeros,
            zeros,
            mask,
            np.array(["a", "b"]),
            np.array([0, 0]),
            np.zeros(2, dtype=bool),
            trajectory_balance=False,
        )
        self.assertTrue(torch.equal(shaped, advantages))
        self.assertTrue(torch.equal(tensors["action_weight"], torch.ones(2)))

    def test_one_unparsed_action_makes_whole_trajectory_abstain(self):
        from agent_system.taspo.credit import build_taspo_advantages

        advantages = torch.ones(3, 2)
        student = torch.zeros_like(advantages)
        teacher = torch.tensor([[0.5, 0.5], [0.8, 0.8], [-0.4, -0.4]])
        response_mask = torch.ones_like(advantages)
        action_mask = response_mask.clone()
        action_mask[1] = 0
        shaped, tensors, metrics = build_taspo_advantages(
            advantages,
            student,
            teacher,
            response_mask,
            np.array(["t", "t", "t"]),
            np.arange(3),
            np.ones(3, dtype=bool),
            action_mask=action_mask,
            trajectory_balance=False,
        )

        self.assertTrue(torch.equal(shaped, advantages))
        self.assertTrue(torch.equal(tensors["action_weight"], torch.ones(3)))
        self.assertEqual(metrics["taspo/active_pi_trajectory_ratio"], 0.0)

    def test_taspo_style_centers_raw_score_before_tanh(self):
        from agent_system.taspo.credit import build_taspo_advantages

        advantages = -torch.ones(4, 1)
        student = torch.zeros_like(advantages)
        teacher = torch.tensor([[0.6], [0.0], [-0.8], [-0.2]])
        shaped, tensors, _ = build_taspo_advantages(
            advantages,
            student,
            teacher,
            torch.ones_like(advantages),
            np.array(["t"] * 4),
            np.arange(4),
            np.ones(4, dtype=bool),
            epsilon=0.4,
            temperature=1.0,
            token_gap_clip=0.01,
            trajectory_balance=False,
        )

        raw = teacher.squeeze(-1)
        expected_centered = raw - raw.mean()
        expected_q = torch.tanh(-expected_centered)
        expected_weights = 1.0 + 0.2 * (expected_q - expected_q.mean())
        self.assertTrue(torch.allclose(tensors["centered_step_score"], expected_centered))
        self.assertTrue(torch.allclose(tensors["reward_aligned_score"], expected_q))
        self.assertTrue(torch.allclose(tensors["action_weight"], expected_weights))
        self.assertTrue(torch.all(shaped < 0))

    def test_hierarchical_multiplier_equalizes_trajectory_mass(self):
        from agent_system.taspo.credit import build_taspo_advantages

        # Two real steps in trajectory a, four in b, and two copied padding rows.
        count = 8
        advantages = torch.ones(count, 1)
        zeros = torch.zeros_like(advantages)
        mask = torch.ones_like(advantages)
        traj = np.array(["a", "a", "b", "b", "b", "b", "a", "b"])
        turns = np.array([0, 1, 0, 1, 2, 3, 0, 0])
        padding = np.array([False] * 6 + [True, True])
        _, tensors, _ = build_taspo_advantages(
            advantages,
            zeros,
            zeros,
            mask,
            traj,
            turns,
            np.zeros(count, dtype=bool),
            padding_rows=padding,
            trajectory_balance=True,
        )
        balance = tensors["trajectory_balance"]
        self.assertTrue(torch.allclose(balance[:2], torch.tensor([2.0, 2.0])))
        self.assertTrue(torch.allclose(balance[2:6], torch.ones(4)))
        self.assertTrue(torch.equal(balance[6:], torch.zeros(2)))


if __name__ == "__main__":
    unittest.main()
