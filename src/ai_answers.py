"""AI-assisted answer suggestions for questions entered by the user.

This module uses the providers' HTTPS APIs directly and only the Python standard
library. API keys can come from the environment, a per-user local config file, or
the UI. Locally saved keys stay outside the repository and use restricted file
permissions where the operating system supports them.
"""

import json
import os
import stat
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


class AIAnswerError(Exception):
    """Raised when no configured AI provider can produce a usable answer."""


PROVIDERS = {
    "groq": {
        "name": "Groq",
        "env": "GROQ_API_KEY",
        "model_env": "GROQ_MODEL",
        "model": "llama-3.3-70b-versatile",
        "kind": "openai",
        "url": "https://api.groq.com/openai/v1/chat/completions",
    },
    "gemini": {
        "name": "Gemini",
        "env": "GEMINI_API_KEY",
        "env_aliases": ("GOOGLE_API_KEY",),
        "model_env": "GEMINI_MODEL",
        "model": "gemini-2.5-flash",
        "kind": "gemini",
        "url": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
    },
    "openai": {
        "name": "OpenAI",
        "env": "OPENAI_API_KEY",
        "model_env": "OPENAI_MODEL",
        "model": "gpt-4o-mini",
        "kind": "openai",
        "url": "https://api.openai.com/v1/chat/completions",
    },
}


def ai_config_directory():
    """Return the per-user configuration directory for saved provider keys."""
    override = os.environ.get("KITTY_TOOLS_CONFIG_DIR")
    if override:
        return os.path.abspath(os.path.expanduser(override))

    home = os.path.expanduser("~")
    if os.name == "nt":
        base = os.environ.get("APPDATA") or os.path.join(home, "AppData", "Roaming")
    elif sys.platform == "darwin":
        base = os.path.join(home, "Library", "Application Support")
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(home, ".config")
    return os.path.join(base, "kitty-tools")


def saved_key_file_path():
    """Return the path used for locally saved AI API keys."""
    return os.path.join(ai_config_directory(), "ai_keys.json")


def load_saved_api_keys():
    """Load locally saved API keys, tightening POSIX permissions if necessary."""
    path = saved_key_file_path()
    try:
        if os.path.islink(path):
            return {}
        if os.name != "nt":
            mode = stat.S_IMODE(os.stat(path).st_mode)
            if mode & 0o077:
                os.chmod(path, 0o600)
        with open(path, "r", encoding="utf-8") as key_file:
            payload = json.load(key_file)
    except (OSError, ValueError, TypeError):
        return {}

    if not isinstance(payload, dict):
        return {}
    keys = payload.get("api_keys", {})
    if not isinstance(keys, dict):
        return {}
    return {
        provider: value.strip()
        for provider, value in keys.items()
        if provider in PROVIDERS and isinstance(value, str) and value.strip()
    }


def save_api_keys(api_keys):
    """Merge and securely write API keys to the local per-user config file.

    The file is plain JSON (not encrypted); POSIX files are restricted to the
    current user. Callers should tell users where it is stored and how to remove it.
    """
    keys = load_saved_api_keys()
    for provider, value in (api_keys or {}).items():
        if provider not in PROVIDERS or not isinstance(value, str):
            continue
        value = value.strip()
        if value:
            keys[provider] = value

    directory = ai_config_directory()
    if not os.path.isdir(directory):
        os.makedirs(directory, mode=0o700)
    if os.name != "nt":
        os.chmod(directory, 0o700)

    file_descriptor, temporary_path = tempfile.mkstemp(
        prefix=".ai-keys-", suffix=".tmp", dir=directory
    )
    try:
        if os.name != "nt":
            os.chmod(temporary_path, 0o600)
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as key_file:
            json.dump({"api_keys": keys}, key_file, indent=2)
            key_file.write("\n")
            key_file.flush()
            os.fsync(key_file.fileno())
        os.replace(temporary_path, saved_key_file_path())
        if os.name != "nt":
            os.chmod(saved_key_file_path(), 0o600)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def clear_saved_api_keys():
    """Delete the local AI key file if it exists."""
    path = saved_key_file_path()
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


SYSTEM_PROMPT = """You are a careful multiple-choice question solver. Work out the answer from the question and the listed choices only. Pay close attention to words such as NOT, EXCEPT, and ALL THAT APPLY. If more than one choice is correct, include every correct choice; if exactly one is correct, return just that one. Never add a choice that was not supplied. If uncertain, give your best-supported selection and lower your confidence rather than claiming certainty.

Return only a JSON object with this exact shape:
{"selected": [1], "confidence": 0.0, "explanation": "brief reason"}
The selected values are one-based choice numbers. The selected array must contain one or more valid choice numbers. Confidence is a number from 0 to 1 and is only your estimate, not a guarantee."""



def parse_answer_response(response_text, choices):
    """Parse and validate a model answer, preserving multiple selections.

    Choice numbers in the model response are one-based. Letter labels (A, B, ...)
    and exact choice text are accepted as a fallback for providers that ignore the
    requested JSON format.
    """
    if not isinstance(response_text, str) or not response_text.strip():
        raise AIAnswerError("The AI provider returned an empty answer.")

    text = response_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        # Some models wrap otherwise-valid JSON in a sentence or code fence.
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace < 0 or last_brace <= first_brace:
            raise AIAnswerError("The AI provider did not return a valid answer format.")
        try:
            data = json.loads(text[first_brace:last_brace + 1])
        except (TypeError, ValueError):
            raise AIAnswerError("The AI provider did not return a valid answer format.")

    if not isinstance(data, dict):
        raise AIAnswerError("The AI provider returned an unexpected answer format.")

    selected = data.get("selected")
    if selected is None:
        selected = data.get("selected_choices", data.get("answer_indexes", data.get("answers")))
    if isinstance(selected, (int, str)) and not isinstance(selected, bool):
        selected = [selected]
    if not isinstance(selected, list) or not selected:
        raise AIAnswerError("The AI provider did not select any answer choices.")

    selected_indexes = []
    normalized_choices = [str(choice).strip().casefold() for choice in choices]
    for value in selected:
        index = None
        if isinstance(value, int) and not isinstance(value, bool):
            index = value
        elif isinstance(value, str):
            value = value.strip()
            if value.isdigit():
                index = int(value)
            elif len(value) == 1 and "a" <= value.casefold() <= "z":
                index = ord(value.casefold()) - ord("a") + 1
            else:
                normalized = value.casefold()
                if normalized in normalized_choices:
                    index = normalized_choices.index(normalized) + 1

        if index is None or index < 1 or index > len(choices):
            raise AIAnswerError("The AI provider selected an invalid choice number.")
        if index not in selected_indexes:
            selected_indexes.append(index)

    confidence = data.get("confidence", 0.5)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.5
    if confidence > 1 and confidence <= 100:
        confidence /= 100.0
    confidence = max(0.0, min(1.0, confidence))

    explanation = data.get("explanation", data.get("reasoning", ""))
    if not isinstance(explanation, str):
        explanation = str(explanation)

    return {
        "selected_indexes": selected_indexes,
        "confidence": confidence,
        "explanation": explanation.strip()[:1200],
    }


class AIAnswerClient:
    """Answer a question with a selected provider or an automatic provider race.

    In auto mode, each configured provider is asked in parallel and the first
    valid answer is returned. This favors low latency, but means a request (and
    possible usage charge) may be made to every provider for which a key exists.
    Choose a specific provider to send the question to just one service.
    """

    PROVIDERS = PROVIDERS
    SPEED_ORDER = ("groq", "gemini", "openai")
    REQUEST_TIMEOUT = 15

    def __init__(self, api_keys=None, models=None, timeout=None):
        api_keys = api_keys or {}
        models = models or {}
        saved_keys = load_saved_api_keys()
        self.timeout = timeout or self.REQUEST_TIMEOUT
        self.api_keys = {}
        self.models = {}

        for provider, config in PROVIDERS.items():
            key = api_keys.get(provider)
            if not key:
                key = os.environ.get(config["env"], "")
            if not key:
                for alias in config.get("env_aliases", ()):
                    key = os.environ.get(alias, "")
                    if key:
                        break
            if not key:
                key = saved_keys.get(provider, "")
            self.api_keys[provider] = str(key).strip() if key else ""

            model = models.get(provider) or os.environ.get(config["model_env"])
            self.models[provider] = str(model or config["model"]).strip()

    def available_providers(self):
        """Return provider IDs with API keys available, in default speed order."""
        return [provider for provider in self.SPEED_ORDER if self.api_keys.get(provider)]

    def answer_question(self, question, choices, provider="auto"):
        """Return the fastest valid AI answer (or use one explicitly selected provider)."""
        question = str(question or "").strip()
        if not question:
            raise AIAnswerError("Enter a question before asking the AI.")

        if isinstance(choices, str):
            choices = [line.strip() for line in choices.splitlines() if line.strip()]
        else:
            choices = [str(choice).strip() for choice in (choices or []) if str(choice).strip()]
        if len(choices) < 2:
            raise AIAnswerError("Enter at least two answer choices, one per line.")
        if len(choices) > 50:
            raise AIAnswerError("A maximum of 50 answer choices is supported.")

        provider = str(provider or "auto").strip().lower()
        if provider not in ("auto",) + tuple(PROVIDERS.keys()):
            raise AIAnswerError("Choose Auto, Groq, Gemini, or OpenAI as the provider.")

        if provider != "auto":
            if not self.api_keys.get(provider):
                env_name = PROVIDERS[provider]["env"]
                if provider == "gemini":
                    env_name += " or GOOGLE_API_KEY"
                raise AIAnswerError(
                    "No {0} API key is configured. Enter it in the AI panel or set {1}.".format(
                        PROVIDERS[provider]["name"], env_name
                    )
                )
            outcome = self._timed_provider_call(provider, question, choices)
            if not outcome["ok"]:
                raise AIAnswerError(outcome["error"])
            return self._format_result(provider, outcome, choices)

        available = self.available_providers()
        if not available:
            raise AIAnswerError(
                "No AI API key is configured. Enter one in the AI panel or set "
                "GROQ_API_KEY, GEMINI_API_KEY/GOOGLE_API_KEY, or OPENAI_API_KEY."
            )

        if len(available) == 1:
            provider = available[0]
            outcome = self._timed_provider_call(provider, question, choices)
            if not outcome["ok"]:
                raise AIAnswerError(outcome["error"])
            return self._format_result(provider, outcome, choices)

        executor = ThreadPoolExecutor(max_workers=len(available))
        futures = {}
        for provider_id in available:
            future = executor.submit(self._timed_provider_call, provider_id, question, choices)
            futures[future] = provider_id

        errors = []
        for future in as_completed(futures):
            provider_id = futures[future]
            try:
                outcome = future.result()
            except Exception as exc:
                errors.append("{0}: {1}".format(PROVIDERS[provider_id]["name"], exc))
                continue

            if outcome["ok"]:
                for pending in futures:
                    if pending is not future:
                        pending.cancel()
                # Other in-flight requests may complete in the background; do not
                # hold up the user's answer while waiting on slower providers.
                executor.shutdown(wait=False)
                return self._format_result(provider_id, outcome, choices)
            errors.append(outcome["error"])

        executor.shutdown(wait=True)
        detail = "; ".join(errors) if errors else "No provider returned a usable answer."
        raise AIAnswerError("All configured AI providers failed. " + detail)

    def _timed_provider_call(self, provider, question, choices):
        started = time.monotonic()
        try:
            answer = self._ask_provider(provider, question, choices)
            return {
                "ok": True,
                "answer": answer,
                "latency_seconds": time.monotonic() - started,
            }
        except Exception as exc:
            if isinstance(exc, AIAnswerError):
                message = str(exc)
            else:
                message = "Unexpected {0} error: {1}".format(PROVIDERS[provider]["name"], exc)
            return {
                "ok": False,
                "error": "{0}: {1}".format(PROVIDERS[provider]["name"], message),
                "latency_seconds": time.monotonic() - started,
            }

    def _format_result(self, provider, outcome, choices):
        answer = outcome["answer"]
        indexes = answer["selected_indexes"]
        return {
            "provider": provider,
            "provider_name": PROVIDERS[provider]["name"],
            "model": self.models[provider],
            "latency_ms": int(round(outcome["latency_seconds"] * 1000)),
            "selected_indexes": indexes,
            "selected_answers": [choices[index - 1] for index in indexes],
            "confidence": answer["confidence"],
            "explanation": answer["explanation"],
        }

    def _ask_provider(self, provider, question, choices):
        raw_text = self._request_provider(provider, question, choices)
        return parse_answer_response(raw_text, choices)

    def _request_provider(self, provider, question, choices):
        config = PROVIDERS[provider]
        model = self.models[provider]
        choices_text = [
            {"number": index, "text": choice}
            for index, choice in enumerate(choices, 1)
        ]
        user_prompt = json.dumps(
            {"question": question, "choices": choices_text},
            ensure_ascii=False,
        )

        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if config["kind"] == "openai":
            headers["Authorization"] = "Bearer " + self.api_keys[provider]
            payload = {
                "model": model,
                "temperature": 0,
                "max_tokens": 300,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            }
            url = config["url"]
        else:
            headers["x-goog-api-key"] = self.api_keys[provider]
            payload = {
                "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                "generationConfig": {
                    "temperature": 0,
                    "maxOutputTokens": 300,
                    "responseMimeType": "application/json",
                },
            }
            url = config["url"].format(model=quote(model, safe=""))

        request = Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                response_data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="replace")
                detail = json.loads(body).get("error", {}).get("message", "")
            except Exception:
                detail = ""
            if exc.code == 401 or exc.code == 403:
                detail = detail or "check that the API key is valid and has access to this model"
            elif exc.code == 429:
                detail = detail or "rate limit or quota reached"
            else:
                detail = detail or exc.reason or "request rejected"
            raise AIAnswerError("HTTP {0}: {1}".format(exc.code, detail))
        except URLError as exc:
            raise AIAnswerError("network error: {0}".format(exc.reason))
        except (OSError, ValueError) as exc:
            raise AIAnswerError("request failed: {0}".format(exc))

        if not isinstance(response_data, dict):
            raise AIAnswerError("The provider returned an unexpected response.")
        if response_data.get("error"):
            error = response_data["error"]
            if isinstance(error, dict):
                error = error.get("message", "Provider returned an error.")
            raise AIAnswerError(str(error))

        if config["kind"] == "openai":
            try:
                content = response_data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                raise AIAnswerError("The provider response did not contain an answer.")
            if isinstance(content, list):
                content = "\n".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )
            if not isinstance(content, str):
                raise AIAnswerError("The provider response did not contain text.")
            return content

        try:
            parts = response_data["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError, TypeError):
            raise AIAnswerError("Gemini did not return an answer (it may have been blocked).")
        content = "\n".join(
            part.get("text", "") for part in parts if isinstance(part, dict)
        )
        if not content.strip():
            raise AIAnswerError("Gemini returned an empty answer.")
        return content
