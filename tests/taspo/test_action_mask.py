import unittest

try:
    import torch
except ImportError:
    torch = None


class CharacterTokenizer:
    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in token_ids)


class FakeBatch:
    def __init__(self, texts):
        width = max(len(text) for text in texts)
        responses = []
        masks = []
        for text in texts:
            padding = width - len(text)
            responses.append([ord(char) for char in text] + [0] * padding)
            masks.append([1] * len(text) + [0] * padding)
        self.batch = {
            "responses": torch.tensor(responses),
            "response_mask": torch.tensor(masks),
        }
        self.non_tensor_batch = {}


@unittest.skipIf(torch is None, "PyTorch is required for action-span tests")
class ActionMaskTest(unittest.TestCase):
    def test_masks_only_executable_xml_span(self):
        from agent_system.taspo.action_mask import build_action_token_mask

        text = "<think>look</think><action>take apple</action>tail"
        batch = FakeBatch([text])
        mask, parsed = build_action_token_mask(batch, CharacterTokenizer())
        selected = "".join(
            char
            for char, keep in zip(text, mask[0].tolist())  # noqa: B905 - same width
            if keep
        )
        self.assertEqual(selected, "<action>take apple</action>")
        self.assertTrue(parsed[0].item())

    def test_malformed_turn_abstains_from_privileged_scoring(self):
        from agent_system.taspo.action_mask import build_action_token_mask

        text = "no executable tags"
        batch = FakeBatch([text])
        mask, parsed = build_action_token_mask(batch, CharacterTokenizer())
        self.assertEqual(mask.sum().item(), 0)
        self.assertFalse(parsed[0].item())


if __name__ == "__main__":
    unittest.main()
