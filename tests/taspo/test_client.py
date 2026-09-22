import unittest

from agent_system.taspo.client import (
    AnalyzerClientConfig,
    OpenAIJSONClient,
    parse_json_object,
)


class ClientTest(unittest.TestCase):
    def test_parses_fenced_json_object(self):
        self.assertEqual(parse_json_object('```json\n{"ok": true}\n```'), {"ok": True})

    def test_rejects_missing_glm_endpoint_before_network_setup(self):
        with self.assertRaisesRegex(ValueError, "base URL"):
            OpenAIJSONClient(AnalyzerClientConfig(api_key="test", base_url=""))


if __name__ == "__main__":
    unittest.main()
