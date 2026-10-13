import os
import stat
import sys
import tempfile
import time
import unittest
from unittest.mock import patch


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPOSITORY_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from ai_answers import (
    AIAnswerClient,
    AIAnswerError,
    clear_saved_api_keys,
    load_saved_api_keys,
    parse_answer_response,
    save_api_keys,
    saved_key_file_path,
)


class ParseAnswerResponseTests(unittest.TestCase):
    def test_keeps_all_multiple_correct_choices(self):
        result = parse_answer_response(
            '{"selected": [1, 3, 1], "confidence": 0.9, "explanation": "Both apply."}',
            ["First", "Second", "Third"],
        )
        self.assertEqual(result["selected_indexes"], [1, 3])
        self.assertEqual(result["confidence"], 0.9)
        self.assertEqual(result["explanation"], "Both apply.")

    def test_accepts_letter_answers_and_fenced_json(self):
        result = parse_answer_response(
            '```json\n{"selected": ["B"], "confidence": 80}\n```',
            ["Alpha", "Beta", "Gamma"],
        )
        self.assertEqual(result["selected_indexes"], [2])
        self.assertEqual(result["confidence"], 0.8)

    def test_rejects_out_of_range_choice(self):
        with self.assertRaises(AIAnswerError):
            parse_answer_response('{"selected": [4]}', ["A", "B", "C"])

    def test_requires_at_least_one_selection(self):
        with self.assertRaises(AIAnswerError):
            parse_answer_response('{"selected": []}', ["A", "B"])


class AIAnswerClientTests(unittest.TestCase):
    def test_auto_uses_first_valid_provider_response(self):
        client = AIAnswerClient(api_keys={"groq": "test-key", "gemini": "test-key"})
        calls = []

        def fake_ask(provider, question, choices):
            calls.append(provider)
            if provider == "groq":
                time.sleep(0.04)
            else:
                time.sleep(0.005)
            return {
                "selected_indexes": [2],
                "confidence": 0.75,
                "explanation": "The second choice fits.",
            }

        client._ask_provider = fake_ask
        result = client.answer_question("Pick the best answer", ["No", "Yes"], "auto")
        self.assertEqual(result["provider"], "gemini")
        self.assertEqual(result["selected_answers"], ["Yes"])
        self.assertEqual(set(calls), {"groq", "gemini"})

    def test_auto_falls_back_when_one_provider_fails(self):
        client = AIAnswerClient(api_keys={"groq": "test-key", "openai": "test-key"})

        def fake_ask(provider, question, choices):
            if provider == "groq":
                raise AIAnswerError("temporary provider failure")
            return {
                "selected_indexes": [1],
                "confidence": 0.5,
                "explanation": "Fallback result.",
            }

        client._ask_provider = fake_ask
        result = client.answer_question("Pick one", ["A", "B"], "auto")
        self.assertEqual(result["provider"], "openai")
        self.assertEqual(result["selected_answers"], ["A"])

    def test_explicit_provider_requires_its_own_key(self):
        client = AIAnswerClient(api_keys={"groq": "test-key"})
        with self.assertRaises(AIAnswerError):
            client.answer_question("Pick one", ["A", "B"], "openai")

    def test_requires_question_and_multiple_choices(self):
        client = AIAnswerClient(api_keys={"groq": "test-key"})
        with self.assertRaises(AIAnswerError):
            client.answer_question("", ["A", "B"])
        with self.assertRaises(AIAnswerError):
            client.answer_question("Pick one", ["A"])


class SavedKeyTests(unittest.TestCase):
    def test_keys_are_saved_with_restricted_permissions_and_reloaded(self):
        with tempfile.TemporaryDirectory() as config_dir:
            environment = {
                "KITTY_TOOLS_CONFIG_DIR": config_dir,
                "GROQ_API_KEY": "",
                "GEMINI_API_KEY": "",
                "GOOGLE_API_KEY": "",
                "OPENAI_API_KEY": "",
            }
            with patch.dict(os.environ, environment):
                save_api_keys({"groq": "groq-secret"})
                self.assertEqual(load_saved_api_keys(), {"groq": "groq-secret"})
                self.assertEqual(AIAnswerClient().api_keys["groq"], "groq-secret")

                if os.name != "nt":
                    mode = stat.S_IMODE(os.stat(saved_key_file_path()).st_mode)
                    self.assertEqual(mode, 0o600)

                save_api_keys({"gemini": "gemini-secret"})
                self.assertEqual(
                    load_saved_api_keys(),
                    {"groq": "groq-secret", "gemini": "gemini-secret"},
                )
                clear_saved_api_keys()
                self.assertEqual(load_saved_api_keys(), {})


class ProviderRequestTests(unittest.TestCase):
    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def read(self):
            import json
            return json.dumps(self.payload).encode("utf-8")

    def test_groq_openai_compatible_request(self):
        client = AIAnswerClient(api_keys={"groq": "test-key"})
        payload = {
            "choices": [{"message": {"content": '{"selected": [2], "confidence": 0.8}'}}]
        }
        with patch("ai_answers.urlopen", return_value=self.FakeResponse(payload)) as mocked_urlopen:
            result = client.answer_question("Choose", ["No", "Yes"], provider="groq")

        request = mocked_urlopen.call_args[0][0]
        self.assertEqual(request.full_url, client.PROVIDERS["groq"]["url"])
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        self.assertEqual(result["selected_answers"], ["Yes"])

    def test_gemini_request_uses_key_header(self):
        client = AIAnswerClient(api_keys={"gemini": "test-key"})
        payload = {
            "candidates": [{"content": {"parts": [{"text": '{"selected": [1]}'}]}}]
        }
        with patch("ai_answers.urlopen", return_value=self.FakeResponse(payload)) as mocked_urlopen:
            result = client.answer_question("Choose", ["Yes", "No"], provider="gemini")

        request = mocked_urlopen.call_args[0][0]
        self.assertIn("gemini-2.5-flash:generateContent", request.full_url)
        self.assertEqual(request.headers.get("X-goog-api-key"), "test-key")
        self.assertEqual(result["selected_answers"], ["Yes"])


if __name__ == "__main__":
    unittest.main()
