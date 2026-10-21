

from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import time
from typing import Any, Iterator
from urllib.parse import urlparse
import uuid

from radoncgym.agents.language import (
    LanguageModelResult,
    _candidate_action_json_schema,
    _image_data_url,
    _validated_image_paths,
)
from radoncgym.core.errors import ContractError


QWEN_API_KEY_ENV = "DASHSCOPE_API_KEY"
QWEN_GATE_ROOT_ENV = "RADONCGYM_QWEN_GATE_ROOT"
QWEN_GATE_WAIT_SECONDS_ENV = "RADONCGYM_QWEN_GATE_WAIT_SECONDS"
QWEN_ACTIVE_LIMIT_FILENAME = "active_limit"
QWEN_TOOL_NAME = "submit_radoncgym_structured_output"
QWEN_ACCESS_ROUTE = "alibaba_modelstudio_openai_compatible"
QWEN_ACTION_WIRE_FORMAT = "flat_weight_fields_v1"
QWEN_WEIGHT_FIELD_PREFIX = "weight__"
QWEN_FORMAT_INSTRUCTION_ID = "mandatory_single_function_call_complete_fields_v2"
QWEN_FORMAT_INSTRUCTION = (
    "<QWEN_PROVIDER_FORMAT_CONTRACT>\n"
    "The instruction to return one JSON object must be satisfied by calling "
    f"the single {QWEN_TOOL_NAME} function exactly once. Never place the JSON "
    "in assistant message content and never omit the function call. Populate "
    "every required function argument using its exact native JSON type. This "
    "contract changes only provider serialization, not the requested planning "
    "decision.\n"
    "</QWEN_PROVIDER_FORMAT_CONTRACT>"
)


class QwenModelStudioProvider:
    pass

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        region: str,
        thinking_enabled: bool = True,
        thinking_budget: int | None = 4096,
        request_timeout_seconds: float = 1200.0,
        max_concurrency: int = 8,
        max_transient_attempts: int = 3,
        max_output_tokens: int | None = None,
        streaming: bool = True,
    ) -> None:
        if not model or not region:
            raise ContractError("Qwen provider requires model and region")
        parsed = urlparse(base_url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ContractError("Qwen base_url must be an absolute HTTPS URL")
        if thinking_budget is not None and thinking_budget < 1:
            raise ContractError("Qwen thinking budget must be positive or unset")
        if (
            not math.isfinite(float(request_timeout_seconds))
            or request_timeout_seconds <= 0
        ):
            raise ContractError("Qwen request timeout must be positive")
        if not 1 <= max_concurrency <= 256:
            raise ContractError("Qwen max concurrency must be between 1 and 256")
        if max_transient_attempts < 1:
            raise ContractError("Qwen transient attempts must be positive")
        if max_output_tokens is not None and max_output_tokens < 1:
            raise ContractError(
                "Qwen max output tokens must be positive or unset"
            )
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._region = region
        self._thinking_enabled = bool(thinking_enabled)
        self._thinking_budget = thinking_budget
        self._request_timeout_seconds = float(request_timeout_seconds)
        self._max_concurrency = int(max_concurrency)
        self._max_transient_attempts = int(max_transient_attempts)
        self._max_output_tokens = (
            None if max_output_tokens is None else int(max_output_tokens)
        )
        self._streaming = bool(streaming)

    @property
    def provider_name(self) -> str:
        return "qwen_modelstudio"

    @property
    def requested_model(self) -> str:
        return self._model

    @property
    def checkpoint_sha256(self) -> str | None:
        return None

    @property
    def configuration(self) -> Mapping[str, str]:
        return {
            "access_route": QWEN_ACCESS_ROUTE,
            "api_key_env": QWEN_API_KEY_ENV,
            "base_url": self._base_url,
            "format_instruction_id": QWEN_FORMAT_INSTRUCTION_ID,
            "max_concurrency": str(self._max_concurrency),
            "max_output_tokens": (
                "provider_default"
                if self._max_output_tokens is None
                else str(self._max_output_tokens)
            ),
            "region": self._region,
            "response_repair": "none",
            "stateless": "true",
            "streaming": str(self._streaming).lower(),
            "structured_output": "required_function_call_json_schema",
            "thinking_budget": (
                "unset"
                if self._thinking_budget is None
                else str(self._thinking_budget)
            ),
            "thinking_enabled": str(self._thinking_enabled).lower(),
            "timeout_seconds": _format_float(self._request_timeout_seconds),
            "transient_retry_attempts": str(self._max_transient_attempts),
        }

    def reset(self, *, seed: int) -> None:
        del seed

    def complete(self, *, system: str, user: str) -> LanguageModelResult:
        return self._complete(
            system=system,
            user=user,
            image_paths=(),
            json_schema=_candidate_schema_for_user(user),
        )

    def complete_structured(
        self,
        *,
        system: str,
        user: str,
        json_schema: Mapping[str, Any],
    ) -> LanguageModelResult:
        return self._complete(
            system=system,
            user=user,
            image_paths=(),
            json_schema=json_schema,
        )

    def complete_with_images(
        self,
        *,
        system: str,
        user: str,
        image_paths: tuple[str, ...],
    ) -> LanguageModelResult:
        return self._complete(
            system=system,
            user=user,
            image_paths=image_paths,
            json_schema=_candidate_schema_for_user(user),
        )

    def complete_with_images_structured(
        self,
        *,
        system: str,
        user: str,
        image_paths: tuple[str, ...],
        json_schema: Mapping[str, Any],
    ) -> LanguageModelResult:
        return self._complete(
            system=system,
            user=user,
            image_paths=image_paths,
            json_schema=json_schema,
        )

    def _complete(
        self,
        *,
        system: str,
        user: str,
        image_paths: tuple[str, ...],
        json_schema: Mapping[str, Any],
    ) -> LanguageModelResult:
        schema = _unwrapped_schema(json_schema)
        resolved_images = _validated_image_paths(image_paths)
        image_hashes = tuple(
            hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in resolved_images
        )
        wire_schema, wire_format = _qwen_wire_schema(schema)
        payload = self._request_payload(
            system=system,
            user=user,
            image_paths=resolved_images,
            schema=wire_schema,
        )
        started = time.monotonic()
        gate_wait_seconds = 0.0
        attempts: list[dict[str, Any]] = []
        response: Mapping[str, Any] | None = None
        for attempt_index in range(1, self._max_transient_attempts + 1):
            attempt_started = time.monotonic()
            try:
                with qwen_concurrency_slot(
                    configured_limit=self._max_concurrency
                ) as gate_wait:
                    gate_wait_seconds += gate_wait
                    response = self._send_request(payload)
            except Exception as error:
                status = _exception_status_code(error)
                retryable = _is_transient_error(error, status)
                attempts.append(
                    {
                        "attempt": attempt_index,
                        "error_type": type(error).__name__,
                        "http_status": status,
                        "latency_seconds": time.monotonic() - attempt_started,
                        "status": "retryable_error" if retryable else "fatal_error",
                    }
                )
                if retryable and attempt_index < self._max_transient_attempts:
                    time.sleep(_retry_backoff_seconds(attempt_index))
                    continue
                if isinstance(error, ContractError):
                    raise
                raise ContractError(
                    "Qwen transport failed after "
                    f"{attempt_index} attempt(s): {type(error).__name__}"
                ) from error
            attempts.append(
                {
                    "attempt": attempt_index,
                    "error_type": None,
                    "http_status": 200,
                    "latency_seconds": time.monotonic() - attempt_started,
                    "status": "success",
                }
            )
            break
        if response is None:
            raise AssertionError("Qwen request loop ended without a response")

        message, finish_reason = _response_message(response)
        raw_text, function_call_valid = _function_call_content(
            message,
            wire_format=wire_format,
        )
        reasoning = message.get("reasoning_content")
        reasoning_text = reasoning if isinstance(reasoning, str) else ""
        stream = response.get("_radoncgym_stream")
        stream_metadata = stream if isinstance(stream, Mapping) else {}
        metadata = {
            "access_route": QWEN_ACCESS_ROUTE,
            "action_wire_format": wire_format,
            "base_url_host": urlparse(self._base_url).hostname or "unavailable",
            "finish_reason": finish_reason or "unavailable",
            "function_call_valid": str(function_call_valid).lower(),
            "image_count": str(len(resolved_images)),
            "image_sha256": ",".join(image_hashes),
            "provider_gate_wait_seconds": f"{gate_wait_seconds:.6f}",
            "reasoning_characters": str(len(reasoning_text)),
            "reasoning_content_present": str(bool(reasoning_text)).lower(),
            "region": self._region,
            "response_repair": "none",
            "stream_chunk_count": str(stream_metadata.get("chunk_count", 0)),
            "streaming": str(self._streaming).lower(),
            "time_to_first_chunk_seconds": str(
                stream_metadata.get("time_to_first_chunk_seconds", "unavailable")
            ),
            "tool_name": QWEN_TOOL_NAME,
            "transport_attempts": str(len(attempts)),
            "transport_attempts_json": json.dumps(
                attempts, sort_keys=True, separators=(",", ":")
            ),
            "transport_retries": str(len(attempts) - 1),
        }
        return LanguageModelResult(
            raw_text=raw_text,
            requested_model=self._model,
            response_model=str(response.get("model") or self._model),
            response_id=str(response.get("id") or "unavailable"),
            latency_seconds=time.monotonic() - started,
            usage=tuple(sorted(_normalize_usage(response.get("usage")).items())),
            provider_metadata=tuple(sorted(metadata.items())),
        )

    def _request_payload(
        self,
        *,
        system: str,
        user: str,
        image_paths: tuple[str, ...],
        schema: Mapping[str, Any],
    ) -> dict[str, Any]:
        user_content: str | list[dict[str, Any]] = user
        if image_paths:
            user_content = [{"type": "text", "text": user}]
            user_content.extend(
                {
                    "type": "image_url",
                    "image_url": {"url": _image_data_url(path)},
                }
                for path in image_paths
            )
        messages: list[dict[str, Any]] = []
        raw_required = schema.get("required", [])
        required = (
            tuple(str(item) for item in raw_required)
            if isinstance(raw_required, list)
            else ()
        )
        required_contract = (
            " Required top-level argument names are: "
            + json.dumps(required, separators=(",", ":"))
            + ". Include every listed argument exactly once; an omitted "
            "required argument makes the decision invalid."
            if required
            else ""
        )
        provider_system = (
            f"{system}\n\n{QWEN_FORMAT_INSTRUCTION}{required_contract}"
            if system
            else QWEN_FORMAT_INSTRUCTION + required_contract
        )
        messages.append({"role": "system", "content": provider_system})
        messages.append({"role": "user", "content": user_content})
        extra_body: dict[str, Any] = {
            "enable_thinking": self._thinking_enabled,
        }
        if self._thinking_budget is not None:
            extra_body["thinking_budget"] = self._thinking_budget
        payload: dict[str, Any] = {
            "extra_body": extra_body,
            "messages": messages,
            "model": self._model,
            "n": 1,
            "stream": self._streaming,



            "tool_choice": (
                "auto"
                if self._thinking_enabled
                else {
                    "type": "function",
                    "function": {"name": QWEN_TOOL_NAME},
                }
            ),
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "description": (
                            "Mandatory provider-format boundary: call this "
                            "function exactly once and do not answer in message "
                            "content. Supply the function arguments directly as "
                            "the JSON object required by the schema, using each "
                            "field's exact native JSON type. For a candidate "
                            "action, populate every top-level weight__ numeric "
                            "argument and do not add a nested weights argument. "
                            "Never quote or serialize a nested value. Do not call "
                            "any other tool."
                            + required_contract
                        ),
                        "name": QWEN_TOOL_NAME,
                        "parameters": dict(schema),
                    },
                }
            ],
        }
        if self._max_output_tokens is not None:
            payload["max_completion_tokens"] = self._max_output_tokens
        if self._streaming:
            payload["stream_options"] = {"include_usage": True}
        return payload

    def _send_request(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        api_key = os.environ.get(QWEN_API_KEY_ENV, "")
        if not api_key:
            raise ContractError(
                f"{QWEN_API_KEY_ENV} is required only for a live Qwen request"
            )
        try:
            from openai import OpenAI
        except ImportError as error:
            raise ContractError(
                "Qwen transport requires the optional 'frontier' dependency"
            ) from error
        client = OpenAI(
            api_key=api_key,
            base_url=self._base_url,
            max_retries=0,
            timeout=self._request_timeout_seconds,
        )
        request_started = time.monotonic()
        created = client.chat.completions.create(**dict(payload))
        if self._streaming:
            return aggregate_qwen_tool_stream(
                created,
                request_started_monotonic=request_started,
            )
        value = created.model_dump()
        if not isinstance(value, Mapping):
            raise ContractError("Qwen SDK returned a non-object response")
        return value


def _candidate_schema_for_user(user: str) -> Mapping[str, Any]:
    return _candidate_action_json_schema(
        allow_memory_operations='"memory_management_contract"' in user,
        require_planning_hypothesis='"planning_hypothesis_contract"' in user,
        require_outcome_forecast='"source_aware_forecast_contract"' in user,
    )


def _unwrapped_schema(value: Mapping[str, Any]) -> dict[str, Any]:
    schema = value.get("schema", value)
    if not isinstance(schema, Mapping):
        raise ContractError("Qwen structured output requires a JSON object schema")


    try:
        normalized = json.loads(
            json.dumps(
                schema,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as error:
        raise ContractError(f"Qwen action schema is not JSON: {error}") from error
    if normalized.get("type") != "object":
        raise ContractError("Qwen function parameters must have object type")
    return normalized


def _qwen_wire_schema(
    schema: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    pass

    from radoncgym.core.constants import (
        MAX_OBJECTIVE_WEIGHT,
        MIN_OBJECTIVE_WEIGHT,
        OBJECTIVE_NAMES,
    )

    normalized = json.loads(json.dumps(schema))
    properties = normalized.get("properties")
    if not isinstance(properties, dict):
        return normalized, "native_json_object"
    weights = properties.get("weights")
    if not isinstance(weights, dict):
        return normalized, "native_json_object"
    variants = weights.get("anyOf")
    if not isinstance(variants, list):
        return normalized, "native_json_object"
    object_variant = next(
        (
            variant
            for variant in variants
            if isinstance(variant, Mapping) and variant.get("type") == "object"
        ),
        None,
    )
    if not isinstance(object_variant, Mapping):
        return normalized, "native_json_object"
    weight_properties = object_variant.get("properties")
    if not isinstance(weight_properties, Mapping) or set(weight_properties) != set(
        OBJECTIVE_NAMES
    ):
        return normalized, "native_json_object"
    properties.pop("weights")
    for name in OBJECTIVE_NAMES:
        properties[f"{QWEN_WEIGHT_FIELD_PREFIX}{name}"] = {
            "type": "number",
            "minimum": MIN_OBJECTIVE_WEIGHT,
            "maximum": MAX_OBJECTIVE_WEIGHT,
            "description": f"Candidate objective weight for {name}.",
        }
    required = normalized.get("required")
    if isinstance(required, list):
        normalized["required"] = [item for item in required if item != "weights"]
        normalized["required"].extend(
            f"{QWEN_WEIGHT_FIELD_PREFIX}{name}" for name in OBJECTIVE_NAMES
        )
    return normalized, QWEN_ACTION_WIRE_FORMAT


def aggregate_qwen_tool_stream(
    stream: Iterable[Any],
    *,
    request_started_monotonic: float | None = None,
) -> Mapping[str, Any]:
    pass

    started = (
        time.monotonic()
        if request_started_monotonic is None
        else request_started_monotonic
    )
    first_chunk_seconds: float | None = None
    chunk_count = 0
    request_id: str | None = None
    model: str | None = None
    finish_reason: str | None = None
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: dict[int, dict[str, Any]] = {}
    usage: Mapping[str, Any] = {}
    for chunk in stream:
        if first_chunk_seconds is None:
            first_chunk_seconds = time.monotonic() - started
        chunk_count += 1
        value = chunk.model_dump()
        if not isinstance(value, Mapping):
            raise ContractError("Qwen stream chunk is not an object")
        if value.get("id"):
            request_id = str(value["id"])
        if value.get("model"):
            model = str(value["model"])
        raw_usage = value.get("usage")
        if isinstance(raw_usage, Mapping):
            usage = raw_usage
        choices = value.get("choices")
        if choices in (None, [], ()):
            continue
        if not isinstance(choices, list) or len(choices) != 1:
            raise ContractError("Qwen stream chunk must have at most one choice")
        choice = choices[0]
        if not isinstance(choice, Mapping):
            raise ContractError("Qwen stream choice must be an object")
        if choice.get("finish_reason") is not None:
            finish_reason = str(choice["finish_reason"])
        delta = choice.get("delta")
        if not isinstance(delta, Mapping):
            continue
        content = delta.get("content")
        if isinstance(content, str):
            content_parts.append(content)
        reasoning = delta.get("reasoning_content")
        if isinstance(reasoning, str):
            reasoning_parts.append(reasoning)
        raw_calls = delta.get("tool_calls")
        if not isinstance(raw_calls, list):
            continue
        for raw_call in raw_calls:
            if not isinstance(raw_call, Mapping):
                continue
            index = raw_call.get("index", 0)
            if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                raise ContractError("Qwen streamed tool-call index is invalid")
            record = tool_calls.setdefault(
                index,
                {
                    "id": "",
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                },
            )
            if isinstance(raw_call.get("id"), str):
                record["id"] += raw_call["id"]
            function = raw_call.get("function")
            if isinstance(function, Mapping):
                if isinstance(function.get("name"), str):
                    record["function"]["name"] += function["name"]
                if isinstance(function.get("arguments"), str):
                    record["function"]["arguments"] += function["arguments"]
    if chunk_count == 0:
        raise ContractError("Qwen streaming response contained no chunks")
    return {
        "_radoncgym_stream": {
            "chunk_count": chunk_count,
            "time_to_first_chunk_seconds": first_chunk_seconds,
        },
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {
                    "content": "".join(content_parts),
                    "reasoning_content": "".join(reasoning_parts),
                    "tool_calls": [tool_calls[index] for index in sorted(tool_calls)],
                },
            }
        ],
        "id": request_id or "unavailable",
        "model": model or "unavailable",
        "usage": dict(usage),
    }


def _response_message(
    response: Mapping[str, Any],
) -> tuple[Mapping[str, Any], str | None]:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ContractError("Qwen response must contain exactly one choice")
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise ContractError("Qwen response choice must be an object")
    message = choice.get("message")
    if not isinstance(message, Mapping):
        raise ContractError("Qwen response lacks a message")
    finish = choice.get("finish_reason")
    return message, None if finish is None else str(finish)


def _function_call_content(
    message: Mapping[str, Any],
    *,
    wire_format: str = "native_json_object",
) -> tuple[str, bool]:
    calls = message.get("tool_calls")
    if isinstance(calls, list) and len(calls) == 1:
        call = calls[0]
        function = call.get("function") if isinstance(call, Mapping) else None
        if isinstance(function, Mapping):
            name = function.get("name")
            arguments = function.get("arguments")
            if (
                name == QWEN_TOOL_NAME
                and isinstance(arguments, str)
                and arguments.strip()
            ):
                return _decode_qwen_wire_arguments(
                    arguments,
                    wire_format=wire_format,
                ), True



    fallback = {
        "content": message.get("content"),
        "tool_calls": calls,
    }
    return json.dumps(fallback, sort_keys=True, separators=(",", ":")), False


def _decode_qwen_wire_arguments(arguments: str, *, wire_format: str) -> str:
    if wire_format != QWEN_ACTION_WIRE_FORMAT:
        return arguments
    from radoncgym.core.constants import OBJECTIVE_NAMES

    try:
        value = json.loads(arguments)
    except json.JSONDecodeError:
        return arguments
    if not isinstance(value, dict):
        return arguments
    weights: dict[str, float | int] = {}
    for name in OBJECTIVE_NAMES:
        field = f"{QWEN_WEIGHT_FIELD_PREFIX}{name}"
        item = value.get(field)
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            return arguments
        weights[name] = item
    for name in OBJECTIVE_NAMES:
        value.pop(f"{QWEN_WEIGHT_FIELD_PREFIX}{name}")
    value["weights"] = weights
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _normalize_usage(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, int] = {}
    aliases = (
        ("prompt_tokens", "input_tokens"),
        ("completion_tokens", "output_tokens"),
        ("total_tokens", "total_tokens"),
    )
    for source, target in aliases:
        item = value.get(source)
        if isinstance(item, int) and not isinstance(item, bool) and item >= 0:
            result[target] = item
    prompt_details = value.get("prompt_tokens_details")
    if isinstance(prompt_details, Mapping):
        cached = prompt_details.get("cached_tokens")
        if isinstance(cached, int) and not isinstance(cached, bool) and cached >= 0:
            result["cached_input_tokens"] = cached
    completion_details = value.get("completion_tokens_details")
    if isinstance(completion_details, Mapping):
        reasoning = completion_details.get("reasoning_tokens")
        if (
            isinstance(reasoning, int)
            and not isinstance(reasoning, bool)
            and reasoning >= 0
        ):
            result["reasoning_tokens"] = reasoning
    return result


@contextmanager
def qwen_concurrency_slot(*, configured_limit: int) -> Iterator[float]:
    pass

    root_value = os.environ.get(QWEN_GATE_ROOT_ENV, "").strip()
    if not root_value:
        raise ContractError(
            f"{QWEN_GATE_ROOT_ENV} is required before a live Qwen request"
        )
    gate_root = Path(root_value).expanduser().resolve()
    gate_root.mkdir(parents=True, exist_ok=True)
    queue_root = gate_root / "waiters"
    queue_root.mkdir(parents=True, exist_ok=True)
    try:
        wait_limit = float(
            os.environ.get(QWEN_GATE_WAIT_SECONDS_ENV, "7200")
        )
    except ValueError as error:
        raise ContractError(
            f"{QWEN_GATE_WAIT_SECONDS_ENV} must be numeric"
        ) from error
    if wait_limit <= 0:
        raise ContractError("Qwen gate wait limit must be positive")

    queue_lock = (gate_root / "queue.lock").open("a+")
    ticket_path = queue_root / (
        f"{time.time_ns():020d}-{socket.gethostname()}-{os.getpid()}-"
        f"{uuid.uuid4().hex}.ticket"
    )
    ticket_handle = None
    slot_handle = None
    started = time.monotonic()
    try:
        fcntl.flock(queue_lock.fileno(), fcntl.LOCK_EX)
        try:
            ticket_handle = ticket_path.open("x+")
            fcntl.flock(ticket_handle.fileno(), fcntl.LOCK_EX)
        finally:
            fcntl.flock(queue_lock.fileno(), fcntl.LOCK_UN)
        while True:
            active_limit = configured_limit
            active_path = gate_root / QWEN_ACTIVE_LIMIT_FILENAME
            if active_path.is_file():
                try:
                    active_limit = int(active_path.read_text().strip())
                except ValueError as error:
                    raise ContractError("Qwen active limit is not an integer") from error
                if not 1 <= active_limit <= configured_limit:
                    raise ContractError(
                        "Qwen active limit must be between 1 and its configured ceiling"
                    )
            fcntl.flock(queue_lock.fileno(), fcntl.LOCK_EX)
            try:
                live: list[Path] = []
                for path in sorted(queue_root.glob("*.ticket")):
                    if path == ticket_path:
                        live.append(path)
                        continue
                    handle = path.open("a+")
                    try:
                        fcntl.flock(
                            handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
                        )
                    except BlockingIOError:
                        handle.close()
                        live.append(path)
                        continue
                    path.unlink(missing_ok=True)
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                    handle.close()
                if live and live[0] == ticket_path:
                    for slot in range(active_limit):
                        candidate = (gate_root / f"slot-{slot:03d}.lock").open("a+")
                        try:
                            fcntl.flock(
                                candidate.fileno(),
                                fcntl.LOCK_EX | fcntl.LOCK_NB,
                            )
                        except BlockingIOError:
                            candidate.close()
                            continue
                        slot_handle = candidate
                        ticket_path.unlink(missing_ok=True)
                        break
            finally:
                fcntl.flock(queue_lock.fileno(), fcntl.LOCK_UN)
            if slot_handle is not None:
                yield time.monotonic() - started
                return
            if time.monotonic() - started >= wait_limit:
                raise ContractError("Qwen concurrency-gate wait timed out")
            time.sleep(0.25)
    finally:
        if slot_handle is not None:
            fcntl.flock(slot_handle.fileno(), fcntl.LOCK_UN)
            slot_handle.close()
        if ticket_handle is not None:
            fcntl.flock(queue_lock.fileno(), fcntl.LOCK_EX)
            try:
                ticket_path.unlink(missing_ok=True)
            finally:
                fcntl.flock(queue_lock.fileno(), fcntl.LOCK_UN)
            fcntl.flock(ticket_handle.fileno(), fcntl.LOCK_UN)
            ticket_handle.close()
        queue_lock.close()


def _exception_status_code(error: Exception) -> int | None:
    value = getattr(error, "status_code", None)
    return value if isinstance(value, int) else None


def _is_transient_error(error: Exception, status: int | None) -> bool:
    if status == 429 or (status is not None and 500 <= status <= 599):
        return True
    if isinstance(error, (ConnectionError, TimeoutError)):
        return True
    name = type(error).__name__.lower()
    return "timeout" in name or "connection" in name


def _retry_backoff_seconds(failed_attempt: int) -> float:
    return float(min(2 ** (failed_attempt - 1), 30))


def _format_float(value: float) -> str:
    return format(value, ".17g")


__all__ = [
    "QWEN_ACCESS_ROUTE",
    "QWEN_API_KEY_ENV",
    "QWEN_GATE_ROOT_ENV",
    "QWEN_TOOL_NAME",
    "QwenModelStudioProvider",
    "aggregate_qwen_tool_stream",
    "qwen_concurrency_slot",
]
