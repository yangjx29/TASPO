"""Hardened OpenAI-compatible JSON client for TASPO PI analysis."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests

DEFAULT_ENDPOINT = ""
logger = logging.getLogger(__name__)


def _cfg_get(config, key: str, default=None):
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


def _split_endpoints(value: object) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item for item in re.split(r"[,;\s]+", str(value or "")) if item))


def _message_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""
    chunks: list[str] = []
    for part in value:
        if isinstance(part, str):
            chunks.append(part)
        elif isinstance(part, dict):
            text = part.get("text", part.get("content", ""))
            if isinstance(text, str):
                chunks.append(text)
    return "".join(chunks)


@dataclass(frozen=True)
class AnalyzerClientConfig:
    model: str = "glm-5.2"
    base_url: str = ""
    endpoints: tuple[str, ...] = ()
    api_key: str = ""
    timeout: float = 120.0
    max_retries: int = 1
    temperature: float = 0.0
    max_tokens: int = 4096
    enable_thinking: bool | None = False
    response_format: str = "none"
    cache_dir: str = "outputs/taspo_cache"
    diagnostics_dir: str = "outputs/taspo_diagnostics"
    diagnostics_excerpt_chars: int = 1000
    cache_version: str = "hardened-v1"

    @classmethod
    def from_config(cls, config) -> "AnalyzerClientConfig":
        base_url = os.getenv(
            "TASPO_ANALYZER_BASE_URL",
            os.getenv("OPENAI_BASE_URL", _cfg_get(config, "base_url", DEFAULT_ENDPOINT)),
        )
        endpoints = _split_endpoints(
            os.getenv("TASPO_ANALYZER_ENDPOINTS", _cfg_get(config, "endpoints", base_url))
        )
        return cls(
            model=os.getenv("TASPO_ANALYZER_MODEL", _cfg_get(config, "model", "glm-5.2")),
            base_url=endpoints[0] if endpoints else str(base_url),
            endpoints=endpoints,
            api_key=os.getenv("TASPO_ANALYZER_API_KEY", os.getenv("OPENAI_API_KEY", "")).strip(),
            timeout=float(os.getenv("TASPO_ANALYZER_TIMEOUT", _cfg_get(config, "timeout", 120.0))),
            max_retries=int(os.getenv("TASPO_ANALYZER_MAX_RETRIES", _cfg_get(config, "max_retries", 1))),
            temperature=float(_cfg_get(config, "temperature", 0.0)),
            max_tokens=int(os.getenv("TASPO_ANALYZER_MAX_TOKENS", _cfg_get(config, "max_tokens", 4096))),
            enable_thinking=_env_bool("TASPO_ANALYZER_ENABLE_THINKING", bool(_cfg_get(config, "enable_thinking", False))),
            response_format=str(os.getenv("TASPO_ANALYZER_RESPONSE_FORMAT", _cfg_get(config, "response_format", "none"))).lower(),
            cache_dir=str(_cfg_get(config, "cache_dir", "outputs/taspo_cache")),
            diagnostics_dir=str(_cfg_get(config, "diagnostics_dir", "outputs/taspo_diagnostics")),
            diagnostics_excerpt_chars=int(_cfg_get(config, "diagnostics_excerpt_chars", 1000)),
            cache_version=str(_cfg_get(config, "cache_version", "hardened-v1")),
        )


class AnalyzerResponseError(ValueError):
    def __init__(self, category: str, message: str):
        super().__init__(message)
        self.category = category


def parse_json_object(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise AnalyzerResponseError("empty_content", "Analyzer response content is empty")
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines.pop()
        text = "\n".join(lines).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        start = text.find("{")
        if start < 0:
            raise AnalyzerResponseError("invalid_json", str(exc)) from exc
        try:
            value, _ = json.JSONDecoder().raw_decode(text[start:])
        except json.JSONDecodeError as inner:
            category = "truncated_json" if inner.pos >= len(text[start:]) - 2 else "invalid_json"
            raise AnalyzerResponseError(category, str(inner)) from inner
    if not isinstance(value, dict):
        raise AnalyzerResponseError("response_schema", "Analyzer response must be a JSON object")
    return value


class OpenAIJSONClient:
    def __init__(self, config: AnalyzerClientConfig):
        self.config = config
        self._endpoints = config.endpoints or _split_endpoints(config.base_url)
        if not self._endpoints:
            raise ValueError(
                "TASPO analyzer base URL is missing; set TASPO_ANALYZER_BASE_URL "
                "or TASPO_ANALYZER_ENDPOINTS"
            )
        if config.response_format not in {"none", "json_object"}:
            raise ValueError("response_format must be 'none' or 'json_object'")
        self._cache_dir = Path(config.cache_dir).expanduser()
        self._diagnostics_dir = Path(config.diagnostics_dir).expanduser()
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._diagnostics_dir.mkdir(parents=True, exist_ok=True)
        self._diagnostics_path = self._diagnostics_dir / "analyzer_requests.jsonl"
        self._diagnostics_lock = threading.Lock()
        self._usage_lock = threading.Lock()
        self._usage = {key: 0.0 for key in (
            "live_calls", "successful_calls", "failed_calls", "retries", "cache_hits",
            "prompt_tokens", "completion_tokens", "request_seconds", "content_chars",
            "reasoning_chars", "empty_content", "reasoning_only", "length_finishes",
            "timeout_errors", "http_errors", "network_errors", "invalid_json",
            "truncated_json", "response_schema_errors",
        )}

    def snapshot_usage(self) -> dict[str, float]:
        with self._usage_lock:
            return dict(self._usage)

    def complete_json(self, system_prompt: str, payload: dict, namespace: str) -> dict:
        request = {
            "cache_version": self.config.cache_version,
            "model": self.config.model,
            "endpoints": self._endpoints,
            "system": system_prompt,
            "payload": payload,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "thinking": self.config.enable_thinking,
            "response_format": self.config.response_format,
        }
        digest = hashlib.sha256(json.dumps(request, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        cache_path = self._cache_dir / f"{namespace}-{digest}.json"
        cached = self._read_cache(cache_path)
        if cached is not None:
            self._inc(cache_hits=1)
            return cached
        user_content = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            endpoint = self._endpoints[(int(digest[:8], 16) + attempt) % len(self._endpoints)]
            started = time.monotonic()
            metadata: dict[str, Any] = {"finish_reason": "", "content": "", "reasoning": "", "usage": {}}
            try:
                body: dict[str, Any] = {
                    "model": self.config.model,
                    "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}],
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                }
                if self.config.enable_thinking is not None:
                    body["chat_template_kwargs"] = {"enable_thinking": self.config.enable_thinking}
                if self.config.response_format == "json_object":
                    body["response_format"] = {"type": "json_object"}
                headers = {"Content-Type": "application/json", "Connection": "close"}
                if self.config.api_key:
                    headers["Authorization"] = f"Bearer {self.config.api_key}"
                response = requests.post(endpoint, headers=headers, json=body, timeout=self.config.timeout)
                response.raise_for_status()
                data = response.json()
                choice = data["choices"][0]
                message = choice["message"]
                metadata = {
                    "finish_reason": str(choice.get("finish_reason", "") or ""),
                    "content": _message_text(message.get("content", "")),
                    "reasoning": _message_text(message.get("reasoning_content", message.get("reasoning", ""))),
                    "usage": data.get("usage", {}) or {},
                }
                if not metadata["content"].strip() and metadata["reasoning"].strip():
                    raise AnalyzerResponseError("reasoning_only", "Analyzer returned reasoning but no final content")
                parsed = parse_json_object(metadata["content"])
                usage = metadata["usage"]
                increments = {
                    "live_calls": 1, "successful_calls": 1,
                    "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                    "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                    "request_seconds": time.monotonic() - started,
                    "content_chars": len(metadata["content"]), "reasoning_chars": len(metadata["reasoning"]),
                    "length_finishes": float(metadata["finish_reason"] == "length"),
                }
                self._inc(**increments)
                self._write_cache(cache_path, digest, parsed, usage)
                self._diagnose(digest, namespace, endpoint, attempt + 1, "success", "", metadata, time.monotonic() - started, len(user_content))
                return parsed
            except Exception as exc:
                last_error = exc
                category = self._error_category(exc)
                self._inc(
                    live_calls=1, failed_calls=1, retries=float(attempt < self.config.max_retries),
                    request_seconds=time.monotonic() - started, content_chars=len(metadata["content"]),
                    reasoning_chars=len(metadata["reasoning"]), empty_content=float(category == "empty_content"),
                    reasoning_only=float(category == "reasoning_only"), invalid_json=float(category == "invalid_json"),
                    truncated_json=float(category == "truncated_json"), response_schema_errors=float(category == "response_schema"),
                    timeout_errors=float(category == "timeout"), http_errors=float(category == "http_error"), network_errors=float(category == "network_error"),
                    length_finishes=float(metadata["finish_reason"] == "length"),
                )
                self._diagnose(digest, namespace, endpoint, attempt + 1, "failure", category, metadata, time.monotonic() - started, len(user_content))
                logger.warning("TASPO analyzer request failed namespace=%s endpoint=%s attempt=%d category=%s: %s", namespace, endpoint, attempt + 1, category, exc)
                if attempt < self.config.max_retries:
                    time.sleep((1.0, 5.0, 15.0)[min(attempt, 2)])
        raise RuntimeError(f"Analyzer request failed after {self.config.max_retries + 1} attempts: {last_error}") from last_error

    def _inc(self, **values: float) -> None:
        with self._usage_lock:
            for key, value in values.items():
                self._usage[key] = self._usage.get(key, 0.0) + value

    def _diagnose(self, digest: str, namespace: str, endpoint: str, attempt: int, status: str, category: str, metadata: dict[str, Any], duration: float, payload_chars: int) -> None:
        limit = max(0, self.config.diagnostics_excerpt_chars)
        row = {
            "request_hash": digest, "namespace": namespace, "endpoint": endpoint, "attempt": attempt,
            "status": status, "error_category": category, "duration_seconds": round(duration, 6),
            "payload_chars": payload_chars, "finish_reason": metadata.get("finish_reason", ""),
            "prompt_tokens": int(metadata.get("usage", {}).get("prompt_tokens", 0) or 0),
            "completion_tokens": int(metadata.get("usage", {}).get("completion_tokens", 0) or 0),
            "content_chars": len(metadata.get("content", "")), "reasoning_chars": len(metadata.get("reasoning", "")),
            "content_excerpt": metadata.get("content", "")[:limit], "reasoning_excerpt": metadata.get("reasoning", "")[:limit],
        }
        with self._diagnostics_lock:
            with self._diagnostics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    @staticmethod
    def _error_category(exc: Exception) -> str:
        if isinstance(exc, AnalyzerResponseError):
            return exc.category
        if isinstance(exc, requests.Timeout):
            return "timeout"
        if isinstance(exc, requests.HTTPError):
            return "http_error"
        if isinstance(exc, requests.RequestException):
            return "network_error"
        if isinstance(exc, json.JSONDecodeError):
            return "transport_json_error"
        return "unknown"

    @staticmethod
    def _read_cache(path: Path) -> dict | None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data["response"] if isinstance(data.get("response"), dict) else None
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            path.unlink(missing_ok=True)
            return None

    @staticmethod
    def _write_cache(path: Path, digest: str, response: dict, usage: dict) -> None:
        temporary = path.with_suffix(f".{threading.get_ident()}.tmp")
        temporary.write_text(json.dumps({"request_hash": digest, "response": response, "usage": usage}, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)

    def config_record(self) -> dict[str, Any]:
        record = asdict(self.config)
        record["api_key"] = "<redacted>" if record["api_key"] else ""
        return record
