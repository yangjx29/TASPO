import importlib
import sys
import types
import unittest

try:
    import torch
except ImportError:
    torch = None


class CharacterTokenizer:
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(char) for char in text]


class FakeDataProto:
    def __init__(self, tensors, non_tensors=None, meta_info=None):
        self.batch = tensors
        self.non_tensor_batch = non_tensors or {}
        self.meta_info = meta_info or {}

    @classmethod
    def from_dict(cls, tensors=None, non_tensors=None, meta_info=None, **_kwargs):
        return cls(tensors, non_tensors, meta_info)


def import_teacher_module():
    fake_verl = types.ModuleType("verl")
    fake_verl.DataProto = FakeDataProto
    fake_utils = types.ModuleType("verl.utils")
    fake_model = types.ModuleType("verl.utils.model")

    def compute_position_id_with_mask(mask):
        return (torch.cumsum(mask, dim=-1) - 1).clamp_min(0)

    fake_model.compute_position_id_with_mask = compute_position_id_with_mask
    saved = {name: sys.modules.get(name) for name in ["verl", "verl.utils", "verl.utils.model"]}
    sys.modules["verl"] = fake_verl
    sys.modules["verl.utils"] = fake_utils
    sys.modules["verl.utils.model"] = fake_model
    try:
        sys.modules.pop("agent_system.taspo.teacher", None)
        return importlib.import_module("agent_system.taspo.teacher")
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


@unittest.skipIf(torch is None, "PyTorch is required for teacher-batch tests")
class TeacherBatchTest(unittest.TestCase):
    def test_pi_uses_a_dedicated_budget_without_evicting_clean_prompt(self):
        teacher = import_teacher_module()
        # Four prompt slots followed by two response slots.
        batch = FakeDataProto(
            {
                "input_ids": torch.tensor([[0, 0, 65, 66, 70, 71]]),
                "attention_mask": torch.tensor([[0, 0, 1, 1, 1, 1]]),
                "responses": torch.tensor([[70, 71]]),
            },
            {"taspo_teacher_pi": ["use apple"]},
            {"temperature": 1.0},
        )
        output, effective = teacher.build_taspo_teacher_batch(
            batch,
            CharacterTokenizer(),
            pi_token_budget=32,
        )

        valid = output.batch["input_ids"][0][output.batch["attention_mask"][0].bool()]
        self.assertTrue(effective[0].item())
        self.assertTrue(torch.equal(valid[-4:], torch.tensor([65, 66, 70, 71])))
        self.assertTrue(torch.equal(output.batch["responses"], batch.batch["responses"]))
        self.assertEqual(output.batch["input_ids"].shape[-1], 38)

    def test_empty_pi_is_marked_ineffective(self):
        teacher = import_teacher_module()
        batch = FakeDataProto(
            {
                "input_ids": torch.tensor([[65, 66, 70]]),
                "attention_mask": torch.ones(1, 3, dtype=torch.long),
                "responses": torch.tensor([[70]]),
            },
            {"taspo_teacher_pi": [""]},
        )
        _, effective = teacher.build_taspo_teacher_batch(
            batch,
            CharacterTokenizer(),
            pi_token_budget=8,
        )
        self.assertFalse(effective[0].item())


if __name__ == "__main__":
    unittest.main()
